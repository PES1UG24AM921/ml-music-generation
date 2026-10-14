New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" -OutFile data\maestro_midi.zip
Expand-Archive -Force data\maestro_midi.zip -DestinationPath data
Write-Host "MAESTRO ready in data/"
