param(
    [int]$Port = 8765,
    [ValidateSet("responses", "chat_completions")]
    [string]$ApiFormat = "responses",
    [switch]$ConfigureModel
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:OPENAI_API_FORMAT = $ApiFormat

if ($ConfigureModel) {
    $env:OPENAI_MODEL = Read-Host "模型名称 (OPENAI_MODEL)"
    $baseUrl = Read-Host "兼容 Base URL（API 根地址，例如 https://provider.example/v1；留空清除旧值）"
    if ([string]::IsNullOrWhiteSpace($baseUrl)) {
        Remove-Item Env:OPENAI_BASE_URL -ErrorAction SilentlyContinue
    } else {
        $env:OPENAI_BASE_URL = $baseUrl.Trim()
    }
    $secureKey = Read-Host "API Key（仅当前进程）" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

Write-Host "Guide Agent 演示页：http://127.0.0.1:$Port/（LangChain / MCP / $ApiFormat）"
uv run --frozen uvicorn guide_agent.demo_api:app --host 127.0.0.1 --port $Port
