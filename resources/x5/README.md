# X5 selected model

This local package contains the X5 model selected by the user for
`RobotLab-X5-MoECTS-v0` on 2026-09-09. The registered task inherits the
`RobotLab-Legbot-v0` terrain, commands, curricula, randomization and algorithm,
with X5-specific asset/control and physical-unit calibration. Flat, zero-command
mechanical checks use separate diagnostic overrides. No PPO training has run.

Source:

`/home/xgy/重要文档/LocoWheeledLegged_X5_migration/LocoWheeledLegged/locowheeledlegged/assets/x5/urdf/x5.urdf`

The selected model is **not** the different copy in
`/home/xgy/下载/sevnce_description_x5/urdf/x5.urdf`.

`manifest.json` records the source location, source URDF checksum and local
checksums for the URDF plus its 13 referenced STL files. Import only adds one
trailing newline to the URDF. All other URDF bytes and every mesh byte are
unchanged. The four unreferenced meshes and `x5_original.urdf` were not copied.
Existing files in the source project were not modified. No asset license or
permission to publish is inferred from this local import.

Model facts:

- Total URDF mass: 115.780333 kg, including the fixed sensor and PTZ/fire payload.
- 23 links before fixed-joint merging; 6 fixed, 12 revolute leg and 4 continuous wheel joints.
- HAA/HFE/KFE effort limits: 300/250/250 Nm; speed limits: 6/6/8 rad/s.
- Wheel effort/speed limits: 30 Nm / 30 rad/s; rolling collision radius: 0.1005 m.
- Reference leg pose `(0, 0.75, -1.5)` rad puts the wheel bottoms about
  0.56753 m below the base. This is a geometric check, not an equilibrium test.

Offline audit (requires NumPy, does not launch Isaac Sim or write outputs):

```bash
python3 scripts/tools/validate_x5_asset.py
python3 -m unittest discover -s tests -p 'test_x5_asset.py' -v
```

Run these commands from the repository root with a Python containing NumPy.
The simulator checks in `scripts/tools/check_x5_sim.py` separately validate
merged inertias, contacts, observation/action order, stance and signed axes.
Passing these does not demonstrate learned locomotion. See
[the migration checkpoint document](../../docs/x5_migration.md).
