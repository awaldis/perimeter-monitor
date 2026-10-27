import fiftyone as fo

# Create a dataset from your directory
dataset = fo.Dataset.from_dir(
#    dataset_dir="/mnt/truenas_main_share/perimeter-monitor/car_2026-01-18T10-54-43_frames",
    dataset_dir="/mnt/truenas_main_share/perimeter-monitor/car_2026-01-19T14-06-56_frames",
    dataset_type=fo.types.YOLOv5Dataset, # Takes images and labels automatically
    name="mail_truck_project",
    overwrite=True,
    #classes=["car", "suv", "pickup_truck", "truck", "usmail_truck", "fedex_truck",
#                "ups_truck", "amazon_truck"
#            ]    
)

#config = fo.AppConfig()
#config.show_confidence = True
#config.show_attributes = True

# Launch the App in your browser
#session = fo.launch_app(dataset)
#session = fo.launch_app(dataset, address="0.0.0.0", port=8080, auto=False)
#session.wait()

dataset.add_dir( dataset_dir="/mnt/truenas_main_share/perimeter-monitor-dataset", dataset_type=fo.types.YOLOv5Dataset, label_field="ground_truth", tags=["original_import", "ground_truth"] )

dataset.add_dir("/mnt/truenas_main_share/perimeter-monitor/2026-01-23", dataset_type=fo.types.YOLOv5Dataset, tags=["staging"])