# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is blocked to the next chosen version and name of the milestone at a given date.
A new section Unreleased is opened then for next dev phase.

## Unreleased

### Added
- Introduce internal core module to structure mesh plugin logic
- Add pipelines/parameters package for pipeline configuration handling
- Restore holes_detection application removed from CARS 1.0.0
- Restore point_cloud_fusion application within mesh plugin

### Changed
- BREAKING: Upgrade dependency to CARS 1.0.0 (drop support for CARS 0.8.x)
- BREAKING: Require Python >= 3.10
- Refactor application interfaces to align with CARS 1.0.0 API
- Update imports after removal of projection.points_cloud_conversion in CARS 1.0.0
- Adapt pc_tif_tools to new CARS internal structure
- Update setup configuration for CARS 1.0.0 compatibility
- Update README and project documentation to reflect new compatibility requirements
- Harmonize EPSG handling in terrain mapping and mesh generation pipeline

### Fixed
- Fix import errors caused by deprecated projection module
- Fix CRS propagation issues during DTM mesh creation
- Fix incorrect EPSG type handling (int vs string)


## 0.1.0 First Official Release (2023-12-20)

### Added
- Add cars_point_cloud_to_mesh features
- Detail the release !!