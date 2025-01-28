import glob
from loguru import logger

from config.config_gp import config as config_gp
from models.backward_induction_utils import set_global_cholesky_jitter, test_single_model_gp


def main():
    """
    Main entry point for testing GP models across multiple checkpoint directories.
    """
    # Set global cholesky jitter for numerical stability, if needed
    set_global_cholesky_jitter(config_gp["JITTER_VAL"])

    root_dir = config_gp["ROOT_DIR"]
    subproject = config_gp["SUBPROJECT"]
    proj_path = f"{root_dir}logs/{subproject}/checkpoints/"
    glob_paths = glob.glob(proj_path + "*")

    logger.info(f"Found {len(glob_paths)} model directories in: {proj_path}")

    cnt_model = 0
    for path_modeldir in glob_paths:
        cnt_model = test_single_model_gp(path_modeldir, cnt_model, config_gp)
        cnt_model += 1

    logger.success("All GP model testing is complete.")


if __name__ == "__main__":
    main()
