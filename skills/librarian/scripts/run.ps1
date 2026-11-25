# Native Windows launcher; no Bash, symlinks, or Administrator privileges needed.
$ErrorActionPreference = "Stop"
if ($args.Count -eq 0 -or $args[0] -ne "search") {
    throw 'usage: run.ps1 search --provider <claude|codex|antigravity> "<question>"'
}
$scriptArgs = @($args | Select-Object -Skip 1)
$skillDir = Split-Path $PSScriptRoot -Parent
$skillItem = Get-Item -LiteralPath $skillDir -Force
if ($skillItem.Target) {
    $target = @($skillItem.Target)[0]
    if (-not [IO.Path]::IsPathRooted($target)) {
        $target = Join-Path (Split-Path $skillDir -Parent) $target
    }
    $skillDir = [IO.Path]::GetFullPath($target)
}
$scriptDir = Join-Path $skillDir "scripts"
$root = [IO.Path]::GetFullPath((Join-Path $skillDir "../.."))
$env:PYTHONPATH = $root + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# PowerShell 5 must also decode native output as UTF-8 before forwarding it.
$OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
$python = $env:LIBRARIAN_PYTHON
if (-not $python) {
    $env:PATH += [IO.Path]::PathSeparator + "$HOME/.local/bin" + [IO.Path]::PathSeparator + "$HOME/.cargo/bin"
    $uv = (Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
    if (-not $uv) {
        $uvCache = $env:UV_CACHE_DIR
        if (-not $uvCache) {
            $uvCache = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'uv/cache'
        }
        $uvBinDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath((Join-Path $uvCache 'librarian-bin'))
        $uv = Join-Path $uvBinDir 'uv.exe'
        if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
            [Console]::Error.WriteLine('[Librarian] Python setup: installing uv from astral.sh.')
            $previousInstallDir = $env:UV_UNMANAGED_INSTALL
            try {
                $env:UV_UNMANAGED_INSTALL = $uvBinDir
                $installer = Invoke-RestMethod -UseBasicParsing -TimeoutSec 60 -Uri 'https://astral.sh/uv/0.12.0/install.ps1'
                & ([ScriptBlock]::Create($installer)) | Out-Host
            } finally {
                $env:UV_UNMANAGED_INSTALL = $previousInstallDir
            }
            if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
                throw '[Librarian] Python setup: the uv installer did not produce uv.exe.'
            }
        }
    }
    $previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
    try {
        # The plugin checkout can be read-only. Use a cache environment per lockfile.
        if (-not $previousProjectEnvironment) {
            $cacheDir = & $uv cache dir
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            $lockHash = (Get-FileHash -LiteralPath (Join-Path $root "uv.lock")).Hash
            $env:UV_PROJECT_ENVIRONMENT = Join-Path (Join-Path $cacheDir "librarian") $lockHash
        }

        $syncArgs = @("sync", "--project", $root, "--locked", "--no-dev")
        # Windows PowerShell 5 treats redirected native stderr as ErrorRecords.
        $ErrorActionPreference = "Continue"
        $setup = & $uv @syncArgs 2>&1
        $setupCode = $LASTEXITCODE
        $setupText = ($setup | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
        [Console]::Error.WriteLine($setupText.TrimEnd())
        if ($setupCode -ne 0 -and $setupText -match 'Missing expected target directory for Python minor version link at\s+["'']?([^\r\n"'']+)') {
            $candidate = Join-Path $Matches[1].Trim() "python.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                & $candidate -c "import sys; assert sys.version_info >= (3, 10)" 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    [Console]::Error.WriteLine("[Librarian] Python setup: retrying once with $candidate")
                    & $uv @syncArgs --python $candidate
                    $setupCode = $LASTEXITCODE
                }
            }
        }
        $ErrorActionPreference = "Stop"
        if ($setupCode -ne 0) {
            [Console]::Error.WriteLine("[Librarian] Python setup failed. Use the uv error above and follow SKILL.md setup recovery.")
            exit $setupCode
        }
        & $uv run --no-sync --project $root python (Join-Path $scriptDir "search.py") @scriptArgs
        exit $LASTEXITCODE
    } finally {
        $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
    }
}
& $python (Join-Path $scriptDir "search.py") @scriptArgs
exit $LASTEXITCODE
