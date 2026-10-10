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
    $uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $uv) {
        [Console]::Error.WriteLine("[Librarian] Python setup: need uv. Calling agent: install uv using SKILL.md through the host's normal permission flow and retry, or reuse a provisioned interpreter with LIBRARIAN_PYTHON.")
        exit 127
    }
    # Windows PowerShell 5 treats redirected native stderr as ErrorRecords.
    $ErrorActionPreference = "Continue"
    $setup = & $uv.Source sync --project $root 2>&1
    $setupCode = $LASTEXITCODE
    $setupText = ($setup | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    [Console]::Error.WriteLine($setupText.TrimEnd())
    if ($setupCode -ne 0 -and $setupText -match 'Missing expected target directory for Python minor version link at\s+["'']?([^\r\n"'']+)') {
        $candidate = Join-Path $Matches[1].Trim() "python.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            & $candidate -c "import sys; assert sys.version_info >= (3, 10)" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                [Console]::Error.WriteLine("[Librarian] Python setup: retrying once with $candidate")
                & $uv.Source sync --project $root --python $candidate
                $setupCode = $LASTEXITCODE
            }
        }
    }
    $ErrorActionPreference = "Stop"
    if ($setupCode -ne 0) {
        [Console]::Error.WriteLine("[Librarian] Python setup failed. Calling agent: find a working interpreter and retry with its full path in UV_PYTHON, or reuse a provisioned environment with LIBRARIAN_PYTHON, following SKILL.md recovery. No managed directories were deleted.")
        exit $setupCode
    }
    $python = Join-Path $root ".venv/Scripts/python.exe"
}
& $python (Join-Path $scriptDir "search.py") @scriptArgs
exit $LASTEXITCODE
