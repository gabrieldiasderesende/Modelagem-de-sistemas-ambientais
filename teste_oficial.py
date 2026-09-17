import pylandstats as pls

ls = pls.Landscape("ls100_06.tif")

print(ls.compute_class_metrics_df(
    metrics=[
        "proportion_of_landscape"
    ]
))
