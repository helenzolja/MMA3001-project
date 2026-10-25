"""Smarter HVAC scheduling from indoor sensor data (MMA3001 project).

Pipeline, one module per step:

1. :mod:`mma3001_project.load` - parse the raw export, detect stale sensors,
   clean onto a 10-minute grid.
2. :mod:`mma3001_project.reconstruction` - fill missing readings by
   interpolation and validate the choice of method on synthetic gaps.
3. :mod:`mma3001_project.patterns` - smoothed rates of change and daily
   HVAC switch / occupancy times.
4. :mod:`mma3001_project.thermal` - first-order warm-up and cool-down models
   fitted by least squares.
5. :mod:`mma3001_project.schedule` - simulate on/off schedules and measure
   run-hours, heated degree-hours and comfort.

The notebooks in ``notebooks/`` run these steps in order.
"""