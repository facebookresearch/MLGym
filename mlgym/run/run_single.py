from mlgym.utils.extras import get_devices
from mlgym.agent.base import AgentArguments, BaseAgent
import asyncio
import gymnasium as gym
import logging

class RunSingle:
    """
    Running a single agent?
    """
    
    def __init__(self, args: ScriptArguments):
        """Initialize the Main class with the given arguments."""
        self.args = args
        # ! TODO: Add default hooks and hook initialization here.
        
        logger = get_logger("mlgym-run")
        logging.getLogger("simple_parsing").setLevel(logging.WARNING)
        logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")
        
    def main(self):
        if self.args.gpus_per_agent > 0:
            devices = get_devices() if len(self.args.gpus) == 0 else self.args.gpus
            devices = [str(x) for x in devices]
            
            if self.args.gpus_per_agent > len(devices):
                msg = f"Not enough GPUs available. Required: {self.args.gpus_per_agent}, Available: {len(devices)}"
                raise RuntimeError(msg)
            
            agent_devices = [devices]
            
        else:
            agent_devices = [["cpu_0"]]
        
        # Reset environment
        agent = BaseAgent("primary_0", self.args.agent)
        
        # get the unwrapped environment
        env: MLGymEnv = gym.make(f"mlgym/{self.args.environment.task.id}", devices=agent_devices).unwrapped  # type: ignore
        
        try:
            self.run(agent, env, devices, 0)
            
        except _ContinueLoop:
            pass
        
        except KeyboardInterrupt:
            logger.info("Exiting MLGym environment...")
            env.close()
            
        except SystemExit:
            logger.critical("❌ Exiting because SystemExit was called")
            env.close()
            logger.info("Container closed")
            raise
        
        except Exception as e:
            logger.warning(traceback.format_exc())
            if self.args.raise_exceptions:
                env.close()
                raise e
            if env.task:  # type: ignore
                logger.warning(f"❌ Failed on {env.task_args.id}: {e}")  # type: ignore
            else:
                logger.warning("❌ Failed on unknown instance")
            env.reset_container()
            
        env.close()
        
        
    def run(self, agent: BaseAgent, env: MLGymEnv, devices: list[str], run_idx: int) -> None:
        traj_dir = Path("trajectories") / Path(getuser()) / (self.args.run_name() + f"_run_{run_idx}")
        traj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = traj_dir / f"run-{timestamp}.log"
        logger.info("Logging to %s", log_path)
        add_file_handler(log_path, ["mlgym-run", "MLGym", agent.name, "api_models", "env_utils", "MLGymEnv"])
        if self.args.print_config:
            logger.info(f"📙 Arguments: {self.args.dumps_yaml()}")
        self._save_arguments(traj_dir)

        task_id = self.args.environment.task.id
        # ! TODO: add instance start hooks here

        logger.info("▶️  Beginning task " + str(task_id))

        info = env.reset()
        observation = info.pop("observation")
        if info is None:
            raise _ContinueLoop

        # Get info, task information
        assert isinstance(self.args.environment.task, TaskConfig)
        task = self.args.environment.task.description

        info, trajectory = agent.run(
            env= env,  # type: ignore
            observation=observation,
            traj_dir=traj_dir,
            return_type="info_trajectory",
        )

        logger.info(f"Agent finished running")
        
    def _save_arguments(self, traj_dir: Path):
        """Save the arguments to a yaml file to the run's trajectory directory."""
    
        log_path = traj_dir / "args.yaml"

        if log_path.exists():
            try:
                other_args = self.args.load_yaml(log_path)
                if self.args.dumps_yaml() != other_args.dumps_yaml():  # check yaml equality instead of object equality
                    logger.warning("**************************************************")
                    logger.warning("Found existing args.yaml with different arguments!")
                    logger.warning("**************************************************")
            except Exception as e:
                logger.warning(f"Failed to load existing args.yaml: {e}")

        with log_path.open("w") as f:
            self.args.dump_yaml(f)