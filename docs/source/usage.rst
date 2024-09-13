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
      "inputs": {
        "classification_buildings_description": ["building"],
        "dsm_color": "../clr.tif",
        "point_clouds": {
          "pc1": {
            "x": "epi_pc_X.tif",
            "y": "epi_pc_Y.tif",
            "z": "epi_pc_Z.tif",
            "classification": "epi_classification.tif",
            "color": "epi_pc_color.tif",
            "mask": "epi_pc_msk.tif"
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
        "out_dir": "./out_meshes/",
        "out_epsg": 4978
      }
    }





