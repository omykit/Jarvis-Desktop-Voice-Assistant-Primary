param(
    [switch]$HealthCheck
)

$ErrorActionPreference = "Stop"

function Write-EventJson {
    param(
        [hashtable]$Payload
    )

    $Payload | ConvertTo-Json -Compress
}

try {
    Add-Type -AssemblyName System.Speech

    try {
        $culture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US")
        $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)
    }
    catch {
        $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    }

    if ($HealthCheck) {
        Write-Output (Write-EventJson @{
            type = "status"
            text = "ready"
            culture = if ($null -ne $engine.RecognizerInfo -and $null -ne $engine.RecognizerInfo.Culture) {
                $engine.RecognizerInfo.Culture.Name
            } else {
                ""
            }
        })
        exit 0
    }

    try {
        $dictationGrammar = New-Object System.Speech.Recognition.DictationGrammar
        $dictationGrammar.Name = "JarvisDictation"
        $engine.LoadGrammar($dictationGrammar)
        $engine.SetInputToDefaultAudioDevice()
    }
    catch {
        throw "Windows speech recognition could not access the microphone bridge. You can still use typed commands or the browser voice fallback."
    }

    Write-Output (Write-EventJson @{
        type = "status"
        text = "ready"
        culture = if ($null -ne $engine.RecognizerInfo -and $null -ne $engine.RecognizerInfo.Culture) {
            $engine.RecognizerInfo.Culture.Name
        } else {
            ""
        }
    })

    while ($true) {
        try {
            $result = $engine.Recognize([TimeSpan]::FromSeconds(1))
        }
        catch [System.TimeoutException] {
            continue
        }

        if ($null -eq $result) {
            continue
        }

        $confidence = [Math]::Round($result.Confidence, 2)
        if ($confidence -lt 0.45) {
            continue
        }

        $text = ($result.Text -as [string]).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        Write-Output (Write-EventJson @{
            type = "recognized"
            text = $text
            confidence = $confidence
            grammar = $result.Grammar.Name
        })
    }
}
catch {
    Write-Output (Write-EventJson @{
        type = "error"
        text = $_.Exception.Message
    })
    exit 1
}
finally {
    if ($null -ne $engine) {
        try {
            $engine.Dispose()
        }
        catch {
        }
    }
}
