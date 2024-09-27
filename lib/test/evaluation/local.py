from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/scratch/user/agenuinedream/JointNLT/data/got10k_lmdb'
    settings.got10k_path = '/scratch/user/agenuinedream/JointNLT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = '/scratch/user/agenuinedream/JointNLT/test/analysis/'
    settings.lasot_lmdb_path = '/scratch/user/agenuinedream/JointNLT/data/lasot_lmdb'
    settings.lasot_path = '/scratch/user/agenuinedream/JointNLT/data/LaSOTTest'
    settings.lasottext_path = '/scratch/user/agenuinedream/JointNLT/data/LaSOTText'
    settings.network_path = '/scratch/user/agenuinedream/JointNLT/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/scratch/user/agenuinedream/JointNLT/data/nfs'
    settings.otb_path = '/scratch/user/agenuinedream/JointNLT/data/OTB_sentences'
    settings.prj_dir = '/scratch/user/agenuinedream/JointNLT'
    settings.result_plot_path = '/scratch/user/agenuinedream/JointNLT/test/result_plots'
    settings.results_path = '/scratch/user/agenuinedream/JointNLT/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/scratch/user/agenuinedream/JointNLT'
    settings.segmentation_path = '/scratch/user/agenuinedream/JointNLT/test/segmentation_results'
    settings.tc128_path = '/scratch/user/agenuinedream/JointNLT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/scratch/user/agenuinedream/JointNLT/data/trackingNet'
    settings.uav_path = '/scratch/user/agenuinedream/JointNLT/data/UAV123'
    settings.vot_path = '/scratch/user/agenuinedream/JointNLT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

