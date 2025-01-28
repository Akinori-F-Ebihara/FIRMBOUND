import glob
from loguru import logger
from config.config_cfl import config as config_cfl
from models.backward_induction_utils import test_single_model


def main():
    """
    Main entry point for testing CFL models across multiple directories.
    """
    root_dir = config_cfl["ROOT_DIR"]
    subproject = config_cfl["SUBPROJECT"]
    proj_path = f"{root_dir}logs/{subproject}/checkpoints/"
    glob_paths = glob.glob(proj_path + "*")

    logger.info(f"Found {len(glob_paths)} model directories in: {proj_path}")

    cnt_model = 0
    for path_modeldir in glob_paths:
        cnt_model = test_single_model(path_modeldir, cnt_model, config_cfl)
        cnt_model += 1

    logger.success("ALL testing is complete.")


if __name__ == "__main__":
    main()
