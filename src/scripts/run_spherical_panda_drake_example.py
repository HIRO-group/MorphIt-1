#!/usr/bin/env python

import numpy as np
from pydrake.all import *


def add_robot(plant: MultibodyPlant, urdf):
    (arm_idx,) = Parser(plant).AddModels(urdf)
    return arm_idx


# Discrete time step
dt = 0.05

# Initialize the diagram, plant, and scene graph
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

urdf = "panda_multi_sphere.urdf"
arm_idx = add_robot(plant, urdf)
plant.Finalize()

# Robot position and velocities
q_home = np.array([0.0, -0.185, 0.0, -2.256, 0.0, 2.070, 0.785])
q_home0 = np.hstack((q_home, np.zeros(plant.num_velocities())))

kp = np.array([50.0, 50.0, 50.0, 80.0, 50.0, 10.0, 10.0])
kd = 2.0 * np.sqrt(kp)
ki = np.zeros(7)

print("=== Positions and Velocities")
print("Num actuators: ", plant.num_actuators())
print("Num positions: ", plant.num_positions())
print("Num velocities: ", plant.num_velocities())

controller = InverseDynamicsController(
    robot=plant, kp=kp, ki=ki, kd=kd, has_reference_acceleration=False
)
builder.AddSystem(controller)
desired_state = builder.AddSystem(ConstantVectorSource(q_home0 - q_home0 * 1.0))
builder.Connect(
    plant.get_state_output_port(), controller.get_input_port_estimated_state()
)
builder.Connect(
    desired_state.get_output_port(), controller.get_input_port_desired_state()
)
builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())

# Connect to the visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finalize the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
plant.SetPositionsAndVelocities(plant_context, q_home0)

# Simulate the controller
simulator = Simulator(diagram, diagram_context)
simulator.set_publish_every_time_step(True)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_at_initialization(True)
simulator.Initialize()
simulator.AdvanceTo(5)
