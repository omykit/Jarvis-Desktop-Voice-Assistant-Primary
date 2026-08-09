param(
    [string]$Text = "",
    [int]$Rate = 0,
    [switch]$HealthCheck
)

$ErrorActionPreference = "Stop"

$voice = New-Object -ComObject SAPI.SpVoice
$voice.Rate = $Rate

if ($HealthCheck) {
    Write-Output "ready"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Text)) {
    throw "Text is required unless HealthCheck is used."
}

[void]$voice.Speak($Text)
