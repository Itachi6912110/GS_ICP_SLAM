"""Lockstep single-thread GS-ICP SLAM runner (for gs_stats workload probing).

Runs Tracker and Mapper in ONE process with a deterministic schedule:

    for each frame:
        tracker: GICP registration (+ per-frame GICP stats row)
        mapper : ingest pending keyframe, then --mapping_iters_per_frame
                 training iterations

replacing the two-process, wall-clock-paced execution of gs_icp_slam.py —
required for run-to-run reproducible statistics ("System FPS" is not
meaningful here). Same CLI as gs_icp_slam.py; --arch_stats defaults ON.

    python gs_icp_slam_st.py --dataset_path dataset/Replica/office0 \
        --config configs/Replica/caminfo.txt --output_path <out> --save_results
"""
import sys
from argparse import ArgumentParser

import arch_stats_utils
from gs_icp_slam import GS_ICP_SLAM


def main():
    parser = ArgumentParser(description="lockstep single-thread GS-ICP SLAM")
    parser.add_argument("--dataset_path", default="dataset/Replica/room0")
    parser.add_argument("--config", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action="store_true", default=None)
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    parser.add_argument("--arch_stats", action="store_true", default=True)
    parser.add_argument("--no_arch_stats", dest="arch_stats",
                        action="store_false")
    parser.add_argument("--mapping_iters_per_frame", default=5,
                        help="mapping optimization budget per tracked frame")
    parser.add_argument("--frame_limit", default=None,
                        help="dev-only frame cap; skips the final eval")
    args = parser.parse_args(sys.argv[1:])
    assert not args.verbose, "network viewer is not supported in lockstep mode"

    slam = GS_ICP_SLAM(args)
    arch_stats_utils.init(slam)

    # wire the mapper into the tracker's loop instead of a second process
    slam.tracker.sync_mapper = slam.mapper
    slam.mapper.is_mapping_process_started[0] = 1

    slam.tracker.tracking()   # drives tracking + inline mapping
    if args.frame_limit:
        # capped dev run: eval poses are incomplete, just save the model
        if slam.mapper.save_results:
            import os as _os
            slam.mapper.gaussians.save_ply(
                _os.path.join(slam.mapper.output_path, "scene.ply"))
    else:
        slam.mapper.finalize()  # scene.ply + rendering eval (phase "eval")
    arch_stats_utils.close()


if __name__ == "__main__":
    main()
