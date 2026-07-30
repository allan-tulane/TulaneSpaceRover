"""
persistent_world_map.py
"""

import threading

import numpy as np


class PersistentWorldMemory:
    def __init__(self, grid_size_m=20.0, cell_size_m=0.05, obstacle_cost_threshold=128):
        """
        grid_size_m:   physical width/height of the world-frame grid, in meters.
                       Make this comfortably bigger than the area you expect
                       to cover in the demo run.
        cell_size_m:   should match COSTMAP_RESOLUTION so cells line up with
                       EgoCostMapper's own grid.
        obstacle_cost_threshold: kept for API parity with LingbotMemoryMapper;
                       not used directly since we store the actual cost value,
                       not a boolean, but documents the "hazard" cutoff.
        """
        self.cell_size_m = cell_size_m
        self.grid_px = int(round(grid_size_m / cell_size_m))
        if self.grid_px % 2 == 0:
            self.grid_px += 1
        self.origin_offset_px = self.grid_px // 2
        self.obstacle_cost_threshold = obstacle_cost_threshold

        # -1 = never observed, 0-255 = actual observed cost (same convention
        # as LingbotMemoryMapper, so downstream code doesn't need to change).
        self.world_grid = np.full((self.grid_px, self.grid_px), -1, dtype=np.int16)
        self._grid_lock = threading.Lock()

    # -----------------------------------------------------------------
    # Fill the persistent grid — call this once per cycle, right after
    # EgoCostMapper.generate() gives you a fresh reactive grid.
    # -----------------------------------------------------------------
    def update_from_reactive(self, reactive_grid, reactive_rover_row, reactive_rover_col,
                              reactive_resolution_m, rover_x, rover_y, rover_heading):
        """
        Projects every cell of the current egocentric reactive_grid into
        world coordinates (using odometry) and writes it into the
        persistent world_grid, keeping the worst (highest) cost seen per
        world cell — same fail-safe philosophy as EgoCostMapper itself.
        """
        h, w = reactive_grid.shape
        rows, cols = np.indices((h, w))

        dr = reactive_rover_row - rows
        dc = cols - reactive_rover_col
        local_x = dc * reactive_resolution_m
        local_y = dr * reactive_resolution_m

        cos_h, sin_h = np.cos(rover_heading), np.sin(rover_heading)
        wx = rover_x + local_x * cos_h - local_y * sin_h
        wy = rover_y + local_x * sin_h + local_y * cos_h

        world_cols = (wx / self.cell_size_m + self.origin_offset_px).round().astype(np.int64)
        world_rows = (self.origin_offset_px - wy / self.cell_size_m).round().astype(np.int64)

        in_bounds = ((world_rows >= 0) & (world_rows < self.grid_px)
                     & (world_cols >= 0) & (world_cols < self.grid_px))

        wr = world_rows[in_bounds]
        wc = world_cols[in_bounds]
        cost = reactive_grid[in_bounds]

        with self._grid_lock:
            # Same "keep the worst seen" merge as LingbotMemoryMapper.
            flat_idx = wr.astype(np.int64) * self.grid_px + wc.astype(np.int64)
            order = np.argsort(cost)  # ascending -> last write per index = max
            flat_sorted = flat_idx[order]
            cost_sorted = cost[order]
            grid_flat = self.world_grid.reshape(-1)
            current = grid_flat[flat_sorted]
            take_new = (current == -1) | (cost_sorted > current)
            grid_flat[flat_sorted[take_new]] = cost_sorted[take_new]

    # -----------------------------------------------------------------
    # Same read-back API as LingbotMemoryMapper, unchanged.
    # -----------------------------------------------------------------
    def get_world_cost(self, world_x, world_y):
        col = int(round(world_x / self.cell_size_m + self.origin_offset_px))
        row = int(round(self.origin_offset_px - world_y / self.cell_size_m))
        with self._grid_lock:
            if 0 <= row < self.grid_px and 0 <= col < self.grid_px:
                val = self.world_grid[row, col]
                return None if val == -1 else int(val)
        return None

    def fuse_with_reactive(self, reactive_grid, reactive_rover_row, reactive_rover_col,
                            reactive_resolution_m, rover_x, rover_y, rover_heading):
        fused = reactive_grid.copy()
        h, w = reactive_grid.shape
        for rr in range(h):
            for cc in range(w):
                dr = reactive_rover_row - rr
                dc = cc - reactive_rover_col
                local_x = dc * reactive_resolution_m
                local_y = dr * reactive_resolution_m
                wx = rover_x + local_x * np.cos(rover_heading) - local_y * np.sin(rover_heading)
                wy = rover_y + local_x * np.sin(rover_heading) + local_y * np.cos(rover_heading)
                mem_cost = self.get_world_cost(wx, wy)
                if mem_cost is not None:
                    fused[rr, cc] = max(fused[rr, cc], mem_cost)
        return fused

    def debug_save_png(self, path):
        """Same idea as EgoCostMapper.debug_save_png — dump a viewable
        snapshot of the whole persistent map so far."""
        import cv2
        vis_grid = np.where(self.world_grid < 0, 0, self.world_grid).astype(np.uint8)
        vis = cv2.applyColorMap(vis_grid, cv2.COLORMAP_JET)
        unseen = self.world_grid < 0
        vis[unseen] = (40, 40, 40)
        cv2.drawMarker(vis, (self.origin_offset_px, self.origin_offset_px),
                        (255, 255, 255), markerType=cv2.MARKER_TRIANGLE_UP,
                        markerSize=10, thickness=2)
        cv2.imwrite(path, vis)
