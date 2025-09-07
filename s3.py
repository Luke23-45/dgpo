  
    def _randomize_photometrics(self, light_target: np.ndarray):
        """
        Implements an advanced 3-point lighting strategy with material randomization
        to create a more realistic and visually diverse scene.
        """
        # --- Part 1: Randomize Textures ---
        if self.dr_config.table_textures:
            chosen_table_tex = self.np_random.choice(self.dr_config.table_textures)
            if chosen_table_tex in self._dr_mat_ids:
                self.model.geom_matid[self.table_geom_id] = self._dr_mat_ids[chosen_table_tex]

        if self.dr_config.floor_textures:
            chosen_floor_tex = self.np_random.choice(self.dr_config.floor_textures)
            if chosen_floor_tex in self._dr_mat_ids:
                self.model.geom_matid[self.floor_geom_id] = self._dr_mat_ids[chosen_floor_tex]

        # --- Part 2: Randomize Table Material Properties ---
        # This makes textures look more varied (e.g., matte wood vs. polished)
        table_mat_id = self.model.geom_matid[self.table_geom_id]
        # Randomize shininess (how tight the specular highlight is)
        self.model.mat_shininess[table_mat_id] = self.np_random.uniform(20, 50)
        # Randomize specular intensity (how reflective it is)
        spec_intensity = self.np_random.uniform(0.05, 0.25)

        self.model.mat_specular[table_mat_id]  = spec_intensity
        # --- Part 3: Advanced 3-Point Lighting Randomization ---
        # Define a target for the lights to aim at
        target_pos_light = light_target
        # target_pos_light = np.append(self.TABLE_CENTER, 0.4) + self.np_random.uniform(-0.1, 0.1, size=3)
        # Use a shared Kelvin tint for a coherent color temperature
        kelvin_shift = self.np_random.uniform(-800, 800) # Wider range for more variation
        tint = np.array([1.0 + (kelvin_shift / 2500.0), 1.0, 1.0 - (kelvin_shift / 2500.0)])
        tint = np.clip(tint, 0.80, 1.20)
        base_color_temp = self.np_random.uniform(0.85, 1.0)
        base_color = np.array([1.0, base_color_temp, base_color_temp - 0.1])
        base_color = np.clip(base_color, 0.75, 1.0) 
        key_intensity = self.np_random.uniform(0.7, 0.9)
        fill_intensity = self.np_random.uniform(0.3, 0.5)
        back_intensity = self.np_random.uniform(0.2, 0.4)
        # 1. KEY LIGHT (Main Light)
        key_angle = self.np_random.uniform(np.deg2rad(-60), np.deg2rad(60))
        key_radius = self.np_random.uniform(1.2, 1.8)
        key_height = self.np_random.uniform(1.8, 2.5)
        key_pos = np.array([
            self.TABLE_CENTER[0] + key_radius * np.cos(key_angle),
            self.TABLE_CENTER[1] + key_radius * np.sin(key_angle),
            key_height
        ])
        self.model.light_pos[self.light_id] = key_pos
        self.model.light_dir[self.light_id] = self._safe_normalize(target_pos_light - key_pos)
        self.model.light_diffuse[self.light_id] = key_intensity * base_color
        # Add a small ambient component to lift shadows
        self.model.light_ambient[self.light_id] = self.np_random.uniform(0.1, 0.15, 3)

        # 2. FILL LIGHT
        # Place it offset from the key light to fill in shadows
        fill_angle_offset = self.np_random.uniform(np.deg2rad(90), np.deg2rad(140)) * self.np_random.choice([-1, 1])
        fill_angle = key_angle + fill_angle_offset
        fill_radius = self.np_random.uniform(1.0, 1.5)
        fill_height = self.np_random.uniform(1.5, 2.2)
        fill_pos = np.array([
            self.TABLE_CENTER[0] + fill_radius * np.cos(fill_angle),
            self.TABLE_CENTER[1] + fill_radius * np.sin(fill_angle),
            fill_height
        ])
        fill_tint = np.array([0.85, 0.9, 1.0])
        self.model.light_pos[self.fill_light_id] = fill_pos
        self.model.light_dir[self.fill_light_id] = self._safe_normalize(target_pos_light - fill_pos)
        # Fill light is weaker than the key light
        self.model.light_diffuse[self.fill_light_id] = fill_intensity * base_color * fill_tint
        self.model.light_ambient[self.fill_light_id] = self.np_random.uniform(0.05, 0.1, 3)

        # 3. BACK LIGHT (Rim Light)
        # Place it behind the action to create highlights and separation
        back_angle = key_angle + self.np_random.uniform(np.deg2rad(150), np.deg2rad(210))
        back_radius = self.np_random.uniform(1.5, 2.0)
        back_height = self.np_random.uniform(1.5, 2.0)
        back_pos = np.array([
            self.TABLE_CENTER[0] + back_radius * np.cos(back_angle),
            self.TABLE_CENTER[1] + back_radius * np.sin(back_angle),
            back_height
        ])


        self.model.light_pos[self.back_light_id] = back_pos
        self.model.light_dir[self.back_light_id] = self._safe_normalize(target_pos_light - back_pos)
        # Back light is subtle
        self.model.light_diffuse[self.back_light_id] = back_intensity * base_color
        self.model.light_ambient[self.back_light_id] = np.zeros(3) # No ambient from the back light

# In envs/panda_env.py -> _randomize_photometrics()
def _randomize_photometrics(self, light_target: np.ndarray):
    # --- Part 1: Textures (COMMENT OUT FOR NOW) ---
    # if self.dr_config.table_textures:
    #     ...
    # if self.dr_config.floor_textures:
    #     ...

    # --- Part 2: Materials (COMMENT OUT FOR NOW) ---
    # table_mat_id = ...
    # self.model.mat_shininess[...] = ...
    # spec_intensity = ...
    # self.model.mat_specular[...] = ...

    # --- Part 3: Lighting ---
    target_pos_light = light_target
    print(f"DEBUG: Lighting target is at {np.round(target_pos_light, 2)}") # VERIFY THIS

    # Use HARDCODED values instead of random ones for now
    key_pos = target_pos_light + np.array([1.5, 0.5, 2.0]) # A fixed key light position
    fill_pos = target_pos_light + np.array([-1.0, -1.0, 1.5]) # A fixed fill light
    back_pos = target_pos_light + np.array([0.0, -1.5, 1.0]) # A fixed back light

    self.model.light_pos[self.light_id] = key_pos
    self.model.light_pos[self.fill_light_id] = fill_pos
    self.model.light_pos[self.back_light_id] = back_pos

    # THIS IS THE KEY PART FOR THIS STEP
    self.model.light_dir[self.light_id] = self._safe_normalize(target_pos_light - key_pos)
    self.model.light_dir[self.fill_light_id] = self._safe_normalize(target_pos_light - fill_pos)
    self.model.light_dir[self.back_light_id] = self._safe_normalize(target_pos_light - back_pos)

    # Print the directions to the console
    print(f"DEBUG: Main Light DIR: {np.round(self.model.light_dir[self.light_id], 2)}")
    print(f"DEBUG: Fill Light DIR: {np.round(self.model.light_dir[self.fill_light_id], 2)}")

    # (Keep light color/intensity logic COMMENTED OUT)
    # tint = ...
    # key_intensity_mult = ...
    # self.model.light_diffuse[...] = ...