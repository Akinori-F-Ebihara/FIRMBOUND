import glob
from loguru import logger

from config.config_gp import config as config_gp
from models.backward_induction_utils import train_single_model_gp


def main():
    """
    Main entry point for training GP models on all checkpoint directories.
    """
    root_dir = config_gp["ROOT_DIR"]
    subproject = config_gp["SUBPROJECT"]
    proj_path = f"{root_dir}logs/{subproject}/checkpoints/"
    glob_paths = glob.glob(proj_path + "*")

    if not glob_paths:
        logger.error("No target checkpoints are found! Check your path or subproject name.")
        return

    logger.info(f"Found {len(glob_paths)} model directories in: {proj_path}")

    cnt_model = 0
    for path_modeldir in glob_paths:
        cnt_model = train_single_model_gp(path_modeldir, cnt_model, config_gp)
        cnt_model += 1

    logger.success("Training complete for all GP models.")


if __name__ == "__main__":
    main()
