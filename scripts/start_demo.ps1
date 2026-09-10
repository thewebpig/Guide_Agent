param(
    [int]$Port = 8765,
    [string]$ApiKey,
    [Alias("ConfigureModel")]
    [switch]$ConfigureSecret
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $env:OPENAI_API_KEY = $ApiKey.Trim()
} elseif ($ConfigureSecret) {
    $secureKey = Read-Host "API Key（仅当前进程）" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

uv run --frozen python scripts/validate_config.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "启动已停止，请修正配置后重试。"
    return
}

Write-Host "合肥工业大学工程管理与智能制造研究中心导览：http://127.0.0.1:$Port/"
uv run --frozen uvicorn guide_agent.demo_api:app --host 127.0.0.1 --port $Port
