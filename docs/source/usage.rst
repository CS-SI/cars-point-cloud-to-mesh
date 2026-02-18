=====
Usage
=====


To use it:

.. code-block:: console

  $ cars ./config.json

With defaut configuration:

.. code-block:: json

  {
    "pipeline" : "point_cloud_to_mesh",
    "input": {
      "classification_buildings_description": ["1"],
      "dsm_color": "dsm/image.tif",
      "depth_map": {
        "dm1": {
          "x": "depth_map/left_right/X.tif",
          "y": "depth_map/left_right/Y.tif",
          "z": "depth_map/left_right/Z.tif",
          "classification": "depth_map/left_right/classification.tif",
          "image": "depth_map/left_right/image.tif"
        }
      }
    },    
    "applications": {
      "create_dtm_mesh": {
        "method": "percentile_of_unclassified_points"
      },
      "point_clouds_and_polygons_to_mesh": {
        "method": "alpha_shape_delaunay_dtm_projection",
        "out_mesh_mode": "texture"
      }
    },    
    "output": {
      "directory": "out_meshes/",
      "epsg": 4978
    }
  }





