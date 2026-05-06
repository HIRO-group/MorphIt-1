"""
Robot-mode pipeline for MorphIt.

Three stages, each independently runnable as a CLI module:

    python -m scripts.robot.discover            inspect a robot folder
    python -m scripts.robot.pack_robot_meshes   pack collision meshes -> JSON
    python -m scripts.robot.create_robot_urdf   assemble spherical URDF
    python -m scripts.robot.run_pipeline        all of the above

Each module also exposes its core function as a library import so the
FastAPI service can drive the pipeline without shelling out.
"""
