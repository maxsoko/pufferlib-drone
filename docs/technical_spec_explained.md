# AI Grand Prix Technical Spec Explained

This document explains the main technical terms from `260318_Technical_Spec_0001.pdf` in plain language.

## Big Picture

The technical specification describes a virtual drone race simulator. Contestant software controls a simulated drone by receiving sensor data and sending flight commands. The simulator and the contestant code communicate through MAVLink.

Conceptually:

```text
Simulator world + drone physics -> telemetry/vision -> AI controller -> control commands -> simulator moves drone
```

## Core Terms

`Technical specification`: A document that defines the rules, interfaces, and requirements software must follow.

`Contestant control software`: The code written by a team to autonomously fly the drone.

`Autonomous`: The drone must fly by itself, without human input during the submitted run.

`Simulator`: A virtual environment that models the drone, racecourse, physics, camera, and telemetry.

`Interface`: The agreed way two systems talk to each other. In this spec, the main interface is MAVLink over UDP.

`Telemetry`: Data sent from the simulator to the contestant code, such as attitude, velocity, IMU readings, status, and timing.

`Control commands`: Messages the contestant code sends back to tell the drone what to do.

## Simulation Terms

`High-fidelity`: The simulator tries to behave realistically rather than using an overly simple model.

`Real-time physics`: The simulation runs continuously like the real world, updating many times per second.

`Rigid-body drone flight model`: The drone is modeled as a physical object with position, rotation, mass, forces, and collisions.

`Thrust generation`: The simulated propellers produce force to move the drone.

`Aerodynamic drag`: Air resistance that slows the drone down.

`Gravity`: The downward force pulling the drone.

`Collision physics`: What happens when the drone hits gates, obstacles, boundaries, or terrain.

`120 Hz`: The simulator updates physics 120 times per second. `Hz` means "times per second."

`Deterministic`: Given the same inputs, the environment behaves the same way for every team. This keeps the competition fair.

## Coordinate Terms

`Local Cartesian coordinate system`: A local 3D coordinate system using x/y/z-style positions, not Earth latitude/longitude.

`Geographic coordinates`: GPS-style coordinates such as latitude and longitude.

`GPS simulation is not available`: Contestant code should not expect GPS data.

`Absolute global position is not exposed`: The simulator may know where the drone is globally, but contestant code does not receive that directly.

`LOCAL_NED`: A common drone coordinate frame meaning North, East, Down. In MAVLink, `SET_POSITION_TARGET_LOCAL_NED` means setting a target position, velocity, or acceleration in the local North-East-Down frame.

## Vision Terms

`Forward-facing first-person camera`: A camera mounted as if looking out from the front of the drone.

`Vision stream`: Image or video data sent from the simulator to the AI controller. The current spec says detailed vision stream parameters will be provided separately.

`Perception`: The part of an AI system that interprets camera and sensor data, such as detecting gates or obstacles.

`Planning`: The part that decides where the drone should go next.

`Control`: The part that converts a plan into actual flight commands.

## MAVLink Terms

`MAVLink`: Micro Air Vehicle Link. It is a lightweight message protocol used by drones, flight controllers, simulators, and ground-control software.

`MAVLink v2`: Version 2 of the MAVLink protocol.

`MAVSDK`: A developer library that makes MAVLink easier to use from code.

`MAVSDK-compatible interface`: The simulator should be usable with MAVSDK-style client code.

`UDP`: A fast network protocol. It sends packets without the heavier reliability guarantees of TCP, which is useful for real-time control where low latency matters.

`Transport`: The underlying way messages are carried. In this spec, the transport is UDP.

`Client`: The contestant software.

`Simulator -> Client`: Data flowing from the simulator to the contestant software.

`Client -> Simulator`: Commands flowing from the contestant software to the simulator.

## MAVLink Messages

`HEARTBEAT`: A regular "I am alive and connected" message.

`ATTITUDE`: Drone orientation data, such as roll, pitch, and yaw.

`HIGHRES_IMU`: High-resolution inertial sensor data. IMU means Inertial Measurement Unit, usually accelerometer and gyroscope data.

`SET_POSITION_TARGET_LOCAL_NED`: A command telling the simulator a desired local position, velocity, or acceleration target.

`SET_ATTITUDE_TARGET`: A command telling the simulator a desired orientation or angular behavior.

`TIMESYNC`: Timing synchronization between systems.

`ODOMETRY`: Motion or position estimate data, usually including pose and velocity.

## Runtime Terms

`Runtime environment`: The software environment where contestant code runs.

`Python-based runtime`: The spec expects Python to work, specifically mentioning Python `3.14.2`.

`Endpoint`: The simulator network address and port that a client connects to.

`Stream`: A continuous flow of data, such as a telemetry stream, vision stream, or command stream.

`Low-latency`: Very small delay between sending and receiving data.

`SITL`: Software-in-the-Loop. This means real controller software talks to a simulated drone instead of physical hardware.

`SITL bridge`: The connection layer that lets external AI code exchange telemetry and control commands with the simulator.

## Control Pipeline

The spec describes this conceptual control pipeline:

```text
Vision + Telemetry -> Perception -> Planning -> Control -> Pilot Commands -> Stabilized Controller
```

That means the software receives camera and sensor data, understands the scene, decides a route, converts that route into commands, and sends commands to a lower-level controller that stabilizes the drone.

`Stabilized controller`: A controller that handles low-level flight stability, such as keeping the drone from tumbling while following commands.

`Pilot commands`: High-level commands the software sends, similar to what a human pilot would request.

## Competition Terms

`Qualification phase`: Round One, where teams prove their software can navigate the course.

`Racecourse`: The full virtual path through the gates and obstacles.

`Start gate`: The first gate.

`Intermediate gates`: Gates between the start and finish.

`Finish gate`: The final gate.

`Maximum run duration`: The time limit. In this spec, the maximum run duration is 8 minutes.

`Compliance`: Following the spec's rules.

`Disqualification`: Being removed from eligibility, for example if a human interacts during the submitted autonomous flight.
