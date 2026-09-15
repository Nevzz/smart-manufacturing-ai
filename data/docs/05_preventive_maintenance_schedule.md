# Preventive Maintenance Schedule and Sensor Thresholds

## Normal operating ranges

| Parameter | Normal range | Alarm level |
|---|---|---|
| Air temperature | 295–305 K | outside 294–306 K |
| Process temperature | 305–314 K | above 315 K |
| Temperature difference (process − air) | 9.5–11 K | below 8.6 K |
| Rotational speed | 1300–1800 rpm | below 1300 or above 2200 rpm |
| Torque | 30–50 Nm | below 20 or above 60 Nm |
| Tool wear | 0–200 min | above 200 min |
| Mechanical power | 3500–9000 W | outside this band |

## Daily checks (start of shift, about 15 minutes)
- Record air and process temperature; confirm the difference is above 9 K.
- Check coolant level and top up if below the mark.
- Confirm the tool wear counter matches the physical tool in the spindle.
- Listen for abnormal spindle noise during the first cycle.
- Clear chips from the work area and the conveyor.

## Weekly checks (about 1 hour)
- Clean air intake filters and the coolant tank strainer.
- Check coolant concentration with a refractometer.
- Inspect drive belt tension and condition.
- Re-torque fixture clamps to specification.
- Download and review the drive fault log.

## Monthly checks (about 4 hours, machine stopped)
- Clean the heat exchanger fins.
- Lubricate slides and ball screws per the lubrication chart.
- Measure spindle runout and compare with the baseline.
- Verify machine geometry with a test cut and dimensional check.
- Replace coolant if concentration or pH is out of specification.

## Annual overhaul (2–3 days)
- Replace spindle bearings if runout has drifted beyond tolerance.
- Full coolant system flush and refill.
- Electrical panel inspection, including contactor and relay condition.
- Recalibrate all temperature, speed and torque sensors.
- Update the machine baseline used by the predictive model.

## Priority rule for competing work
When more than one alarm is active, address them in this order: heat
dissipation first (it damages the spindle), then overstrain (it damages the
workpiece and fixture), then tool wear (it damages quality), then power
(usually recoverable). Random failures with no assignable sensor cause should
be logged but not acted on individually.

## Model-driven maintenance
The predictive model outputs a failure probability for each machine reading.
Suggested response bands:
- Below 0.3 — no action, continue running.
- 0.3 to 0.6 — inspect at the next planned break.
- Above 0.6 — schedule intervention within the current shift.

These bands should be reviewed quarterly against actual failure outcomes,
because the cost of a missed failure is far higher than the cost of an
unnecessary inspection.
