<#
Train the world models unattended until you stop it.

    powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1

It runs scripts\train_dynamics_ui.py (docs/guides/dynamics.md): the small and
large models side by side on the same batches. Each launch resumes each run's
model_latest.pt on fresh live histories. There is no time
limit: training runs until you press Stop Training or close the window, which
ends this script. If training fails, the window closes and the run is
relaunched from the last checkpoint (at most 200 steps lost). -Until HH:mm adds a deadline. Windows is kept awake while
this script runs; the display may still turn off. Settings live in the trainer
defaults; -TrainerArgs passes extra ones through, e.g. -TrainerArgs "--cooldown
90000 10000" (the end-of-run learning-rate decay, then a normal exit).

At 350 W a batch-16 run once crashed Windows (bugcheck 0x133 after an NVIDIA
driver fault). Every launch logs the GPU's power limit and warns when it is
above -MaxPowerWatts (set it in an admin prompt: nvidia-smi -pl 250), but
never refuses to train over it. While this
script runs, nvidia-smi logs power, temperature, clocks and utilisation once
a second to output\gpu_telemetry.csv.
#>
param(
    [ValidateSet("dynamics", "layered", "vae", "latent")][string]$Run = "dynamics",
    [string]$Until = "",
    [string]$Python = "C:\Python314\python.exe",
    [int]$MaxRestarts = 20,
    [double]$MaxPowerWatts = 235,
    [string]$TrainerArgs = "",
    [string]$Name = ""                  # a second run beside another: its own log, output\<Run>_<Name>_overnight.log
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $root ("output\${Run}" + $(if ($Name) { "_$Name" } else { "" }) + "_overnight.log")
$trainer = @{ "dynamics" = "scripts\train_dynamics_ui.py"; "layered" = "scripts\train_layered.py"; "vae" = "scripts\train_vae_ui.py"; "latent" = "scripts\train_latent_ui.py" }[$Run]

$deadline = $null
if ($Until) {
    $deadline = [datetime]::Today.Add([timespan]::Parse($Until))
    if ($deadline -le (Get-Date)) { $deadline = $deadline.AddDays(1) }
}

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system sleep until this process exits.
Add-Type -Namespace Power -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
[void][Power.Native]::SetThreadExecutionState([uint32]"0x80000001")

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Get-PowerLimit {
    $limit = [double]((nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits) | Select-Object -First 1)
    if ($limit -gt $MaxPowerWatts) {
        Write-Log "warning: GPU power limit is $limit W, above $MaxPowerWatts W; training anyway (to lower it, in an admin prompt: nvidia-smi -pl $MaxPowerWatts)"
    }
    return $limit
}

Write-Log ("$Run run " + $(if ($deadline) { "until $deadline" } else { "until stopped" }))
$restarts = 0
$telemetry = $null
try {
    $telemetry = Start-Process nvidia-smi -NoNewWindow -PassThru -ArgumentList @(
        "--query-gpu=timestamp,power.draw,enforced.power.limit,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used",
        "--format=csv", "-l", "1", "-f", (Join-Path $root "output\gpu_telemetry.csv"))
    while ($true) {
        $watts = Get-PowerLimit
        $seconds = 0
        if ($deadline) {
            $seconds = [int][math]::Floor(($deadline - (Get-Date)).TotalSeconds)
            if ($seconds -lt 120) { break }
        }
        Write-Log ("Launching " + $(if ($seconds) { "for $seconds s" } else { "until stopped" }) + " (restart $restarts, GPU limit $watts W)")
        # cmd appends the trainer's output bytes unchanged (PowerShell 5.1's *>>
        # would write UTF-16 into the log) and passes its exit code through.
        cmd /c "`"$Python`" -u $trainer --seconds $seconds --close-when-done $TrainerArgs >> `"$log`" 2>&1"
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            Write-Log "Training ended normally (Stop Training, window closed or deadline)"
            break
        }
        $restarts++
        if ($restarts -gt $MaxRestarts) {
            Write-Log "Exit code $code; giving up after $MaxRestarts restarts"
            break
        }
        Write-Log "Exit code $code; resuming from the last checkpoint in 30 s"
        Start-Sleep -Seconds 30
    }
}
finally {
    if ($telemetry -and -not $telemetry.HasExited) { Stop-Process -Id $telemetry.Id -Force }
    [void][Power.Native]::SetThreadExecutionState([uint32]"0x80000000")
    Write-Log "Run script finished"
}
