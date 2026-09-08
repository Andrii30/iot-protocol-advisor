# Running the advisor autonomously

## systemd (long-running watcher)

```bash
sudo useradd --system --home /var/lib/iot-protocol-advisor iot-advisor
sudo mkdir -p /opt/iot-protocol-advisor /var/lib/iot-protocol-advisor
sudo chown -R iot-advisor: /opt/iot-protocol-advisor /var/lib/iot-protocol-advisor

# put the project in /opt/iot-protocol-advisor and build its venv:
sudo -u iot-advisor python3 -m venv /opt/iot-protocol-advisor/.venv
sudo -u iot-advisor /opt/iot-protocol-advisor/.venv/bin/pip install -r /opt/iot-protocol-advisor/requirements.txt

sudo cp deploy/systemd/iot-protocol-advisor.service /etc/systemd/system/
sudoedit /etc/systemd/system/iot-protocol-advisor.service   # set DB URL / query
sudo systemctl daemon-reload
sudo systemctl enable --now iot-protocol-advisor
journalctl -u iot-protocol-advisor -f
```

`switches.csv` in `PROTOCOL_ADVISOR_HOME` accumulates every recommended change
with a `checked_at` timestamp.

## cron (periodic one-shot)

```cron
# every 15 minutes; one-shot run, no --watch
*/15 * * * * cd /opt/iot-protocol-advisor && .venv/bin/python -m protocol_advisor \
  --db "postgresql://advisor:secret@localhost/telemetry" \
  --query "SELECT device_id, current_protocol, payload_size, latency, jitter, throughput, packet_loss FROM measurements WHERE ts > now() - interval '1 hour'" \
  --out /var/lib/iot-protocol-advisor/switches.csv >> /var/log/iot-advisor.log 2>&1
```

The model trains once on first run and is cached in `PROTOCOL_ADVISOR_HOME`;
delete `model.pkl` there (or call `Engine().retrain(...)`) to refresh it.
