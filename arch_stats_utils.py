"""Thin GS-ICP-SLAM shim around gs_stats.WorkloadRecorder.

Module singleton, mirroring MonoGS's utils/arch_stats.py. Everything is a
no-op unless --arch_stats is passed (which also requires the lockstep
single-process runner, gs_icp_slam_st.py, for a deterministic stream).

Phases:
  tracking  — render-less GICP rows: inview analog = trackable-target mask
              (the Gaussian subset published to GICP), contrib analog =
              unique matched target Gaussians of this frame's registration.
              Rows carry the scene_version of the publish snapshot.
  mapping   — the mapper's per-iteration render_3 training render.
  eval      — final calc_2d_metric renders.
"""
import os

_recorder = None
_bvh_phases = ("tracking",)
# Incremented on index-permuting ops (prune/create). Appends keep existing
# Gaussian indices stable, so snapshots survive them.
_prune_epoch = 0


def init(slam_args):
    global _recorder
    if not getattr(slam_args, "arch_stats", False):
        return
    try:
        from gs_stats import WorkloadRecorder, is_bvh_available
        from gs_stats.adapters.monogs import json_deepcopy  # reuse helper
    except ImportError:
        print("arch_stats: gs_stats not installed in this env; disabled")
        return
    import diff_gaussian_rasterization as dgr

    dgr.set_stats_mode(True)
    bvh_cfg = {
        "enabled": bool(is_bvh_available()),
        "scale_factor": 3.0,
        "guard_band": 0.0,
        "phases": list(_bvh_phases),
    }
    meta = {
        "app": "GS_ICP_SLAM",
        "dataset_path": getattr(slam_args, "dataset_path", None),
        "single_process": True,
        "lockstep_mapping_iters_per_frame":
            getattr(slam_args, "mapping_iters_per_frame", None),
        "rasterizer_gs_stats_build": bool(dgr.gs_stats_compiled()),
        "tracking_semantics": {
            "n_inview_unique": "trackable target Gaussians published to GICP",
            "n_contrib_unique": "unique matched target Gaussians "
                                "(get_source_correspondence, distance-filtered)",
            "scene_version": "version of the trackable-publish snapshot "
                             "(current version when no prune since publish)",
            "bvh_superset_note": "n_inview_not_bvh_visible is not meaningful "
                                 "for tracking rows: the trackable target is "
                                 "map-wide, not frustum-limited",
        },
    }
    _recorder = WorkloadRecorder(
        out_dir=os.path.join(slam_args.output_path, "arch_stats"),
        meta=meta,
        bvh_cfg=bvh_cfg,
        flush_every=200,
    )


def enabled():
    return _recorder is not None


def record(render_pkg, phase, frame_id, view_id=None, itr=0, model=None,
           camera=None, extra=None):
    if _recorder is None or render_pkg is None:
        return
    _recorder.record(render_pkg, phase, frame_id, view_id=view_id, itr=itr,
                     model=model, camera=camera, extra=extra)


def record_tracking(frame_id, trackable_mask, matched_mask, snapshot_version,
                    n_total, model=None, camera=None, extra=None,
                    masks_are_live=False):
    """Render-less GICP tracking row.

    masks_are_live=True: the masks were padded to the live model's index
    space (no prune since publish) — row uses the current scene_version and
    BVH ancestor counting applies. Otherwise the masks live in the publish
    snapshot and carry its version."""
    if _recorder is None:
        return
    _recorder.record(None, "tracking", frame_id, itr=0, model=model,
                     camera=camera, n_total=n_total,
                     inview_mask=trackable_mask, contrib_mask=matched_mask,
                     scene_version_override=(None if masks_are_live
                                             else snapshot_version),
                     extra=extra)


def scene_version():
    return _recorder.scene_version if _recorder is not None else 0


def notify_scene_change(reason=""):
    global _prune_epoch
    if reason in ("prune", "create"):
        _prune_epoch += 1
    if _recorder is not None:
        _recorder.bump_scene_version(reason)


def prune_epoch():
    return _prune_epoch


def notify_param_change(reason=""):
    if _recorder is not None:
        _recorder.bump_param_version(reason)


def close():
    global _recorder
    if _recorder is not None:
        _recorder.close()
        _recorder = None
