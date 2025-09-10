import random

num_sensors = 10

rssi_readings = {
    f'ED{i+1}': random.randint(-90, -30) for i in range(num_sensors)
}

print("\nRSSI readings:\n")
for sensor, rssi in rssi_readings.items():
    print(f"{sensor}: {rssi} dBm")

max_sensor = max(rssi_readings, key=rssi_readings.get)
max_rssi = rssi_readings[max_sensor]

print("\nCenter of Origin Estimate (Max RSSI):")
print(f"Sensor: {max_sensor}")
print(f"RSSI: {max_rssi} dBm")
