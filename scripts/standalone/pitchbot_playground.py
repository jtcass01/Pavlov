
__author__ = "Jacob Taylor Cassady"
__email__ = "jcassad1@jh.edu"

from omni.isaac.lab.app import AppLauncher

# Parse any command-line arguments specific to the standalone application
from argparse import ArgumentParser

# add argparse arguments
parser = ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from omni.isaac.lab.sim import (SimulationCfg, SimulationContext, GroundPlaneCfg, DomeLightCfg, 
                                MeshSphereCfg, RigidBodyPropertiesCfg, RigidBodyMaterialCfg, 
                                MassPropertiesCfg, PreviewSurfaceCfg, CollisionPropertiesCfg,
                                UsdFileCfg)
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.envs.mdp import (JointEffortActionCfg, reset_scene_to_default, joint_pos_rel, 
                                     joint_vel_rel, last_action)
from omni.isaac.lab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import KINOVA_JACO2_N7S300_CFG


#################################################
#              Scene Definition                 #
#################################################


@configclass
class PitchBotSceneCfg(InteractiveSceneCfg):
    """Configuration for a PitchBot scene."""
    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=GroundPlaneCfg())

    # lights
    dome_light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

    # Table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=1.0, density=10000),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8))
    )

    # articulation
    pitchbot: ArticulationCfg = KINOVA_JACO2_N7S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    pitchbot.init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), rot=(0.0, 0.0, 0.0, 1.0))

    # Baseball
    baseball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Baseball", 
        spawn=MeshSphereCfg(radius=0.073 / 2.,
                            rigid_props=RigidBodyPropertiesCfg(),
                            mass_props=MassPropertiesCfg(mass=1.0, density=710),
                            collision_props=CollisionPropertiesCfg(),
                            visual_material=PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9),
                                                              roughness=0.7,
                                                              metallic=0,
                                                              opacity=1.0),
                            physics_material=RigidBodyMaterialCfg(
                                static_friction=0.6,
                                dynamic_friction=0.5,
                                restitution=0.3
                            )),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.8 + 0.073 / 2.))
    )


#################################################
#               MDP Settings                    #
#################################################


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos: JointEffortActionCfg = \
        JointEffortActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group"""
        # observation terms (order preserved)
        joint_pos: ObservationTermCfg = \
            ObservationTermCfg(func=joint_pos_rel)
        joint_vel: ObservationTermCfg = \
            ObservationTermCfg(func=joint_vel_rel)
        actions: ObservationTermCfg = \
            ObservationTermCfg(func=last_action)
        
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")


#################################################
#          Environment Configuration            #
#################################################


@configclass
class PitchBotEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the PitchBot environment."""
    # Scene settings
    scene: PitchBotSceneCfg = PitchBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control

        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Run the simulation.

    TODO:

    Args:
        sim (SimulationContext): _description_
        scene (InteractiveScene): _description_"""
    robot: ArticulationCfg = scene["pitchbot"]
    baseball: RigidObjectCfg = scene["baseball"]

    sim_dt = sim.get_physics_dt()
    count: int = 0

    while simulation_app.is_running():
        # Reset the simulation
        if count > 2000:
            count = 0

            # Reset entities
            for entity in [robot, baseball]:
                root_state = entity.data.default_root_state.clone()
                entity.write_root_state_to_sim(root_state)
                entity.reset()

                print("----------------------------------------")
                print("[INFO]: Resetting object state...")

        # Apply sim data
        baseball.write_data_to_sim()

        # Step the simulation
        sim.step()

        # Update the buffers
        robot.update(sim_dt)
        baseball.update(sim_dt)
        # scene.update(sim_dt)

        count += 1


def main():
    """"""
    # Load kit helper
    sim_cfg: SimulationCfg = SimulationCfg()
    sim: SimulationContext = SimulationContext(sim_cfg)

    # Set the main camera position
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.0])

    # Design Scene
    scene_cfg: PitchBotSceneCfg = PitchBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene: InteractiveScene = InteractiveScene(scene_cfg)

    # Play the simulation
    sim.reset()

    print("[INFO]: Setup complete...")

    # Run the simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()

    # Close the application
    simulation_app.close()
