import pylandstats as pls

ls = pls.Landscape("ls100_06.tif")

df = ls.compute_class_metrics_df(
    metrics=[
        "proportion_of_landscape",
        "number_of_patches",
        "total_area"
    ]
)

print(df)
