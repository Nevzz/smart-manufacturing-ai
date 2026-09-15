# Power Failure (PWF) and Overstrain Failure (OSF) — Maintenance Guide

## Power Failure (PWF)

### What it is
Mechanical power is torque multiplied by angular velocity. Power failure is
triggered when this value falls outside the usable band — below about 3500 W
or above about 9000 W. Too little power and the tool stalls in the cut; too
much and the drive train is overloaded.

### Warning signs
- Computed power below 3500 W or above 9000 W
- Spindle motor current spiking or tripping the overload relay
- Audible change in spindle note during the cut
- Torque and speed moving in opposite directions

### Root causes
- Feed rate set too aggressively for the spindle rating
- Drive belt slipping, so commanded speed is not delivered
- Spindle motor winding degradation
- Incoming supply voltage sag affecting the drive
- Wrong program loaded for the product variant

### Recommended actions
1. Stop the cycle and check the spindle drive fault log.
2. Verify feed and speed values against the process sheet for that variant.
3. Inspect the drive belt for glazing, cracking and correct tension.
4. Measure spindle motor current under load and compare with the nameplate.
5. Check incoming supply voltage stability at the panel.

---

## Overstrain Failure (OSF)

### What it is
Overstrain failure is driven by the product of tool wear and torque. When this
product exceeds roughly 11,000 minNm for the L variant, 12,000 for M and
13,000 for H, the tool and workpiece are being strained beyond what the setup
can take. Note that the tolerance rises with product quality variant, so the L
variant fails earliest.

### Warning signs
- Tool wear multiplied by torque approaching the variant threshold
- Torque rising while cutting parameters are unchanged
- Chatter marks or dimensional drift on the part
- Visible deflection of the workpiece or fixture

### Root causes
- Worn tooling combined with a heavy cut, the two effects compounding
- Workpiece not clamped rigidly; fixture wear or loose clamps
- Depth of cut increased without reducing feed
- Material hardness above specification for the incoming batch

### Recommended actions
1. Reduce depth of cut or feed rate immediately.
2. Replace the tool if wear is above 150 minutes.
3. Re-torque all fixture clamps to the specified value.
4. Check incoming material hardness certificates for the batch.
5. If the variant is L, review whether the cutting parameters intended for M
   or H have been applied by mistake.

### Preventive measures
- Maintain variant-specific cutting parameter sheets at the machine.
- Alarm at 90 percent of the wear-times-torque threshold, not at 100 percent.
- Inspect and re-torque fixtures at every shift change.

## Typical downtime
PWF: 1–4 hours depending on whether the drive or belt is at fault.
OSF: 30–60 minutes if caught early; several hours if the workpiece or fixture
is damaged.
