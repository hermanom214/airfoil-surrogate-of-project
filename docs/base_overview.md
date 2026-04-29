## Project Overview

### Goal

* Pipeline for airfoil simulations (CFD, automatic generation, stabilitz and data check)
* Machine learning model - predict aerodynamic coefficients, cp, p and U fields

---

### Steps

* Generates airfoil geometries (NACA profiles)
* Converts geometry to STL
* Creates computational mesh - snappyHexMesh (OpenFOAM)
* Runs CFD simulations (simpleFoam, k-omega SST, lowRe)
* Stores results for dataset creation
* ...

---

### Key problems

* Bad mesh near trailing edge (TE)
* Poor data quality for ML

---

### Core pipeline steps

1. **Geometry generation**

   * Generate NACA airfoil points
   * Force sharp trailing edge
   * Export `.dat` and `.stl`

2. **STL processing**

   * Split STL into:

     * `airfoil_main`
     * `airfoil_TE`
   * Keep watertight surface

3. **Meshing (snappyHexMesh)**

   * Base mesh (blockMesh)
   * Surface snapping
   * Local refinement (airfoil + TE)
   * Boundary layers only on main surface

4. **CFD simulation**

   * Solver: simpleFoam
   * Turbulence: k-omega SST
   * Check stability and residuals

5. **Validation**

   * checkMesh
   * inspect velocity and omega fields
   * detect numerical issues

6. **Dataset creation

   * export fields (pressure, velocity)
   * store in structured format (.npz)
   * prepare for ML

---

### Current focus

* Fix mesh quality near trailing edge
* Stabilize simulations
* Ensure consistent and clean data

---

### Final goal

* Reliable CFD → clean dataset → usable ML model
