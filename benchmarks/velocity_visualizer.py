import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("erwin_benchmark.csv")

# Runtime plot
plt.figure()
for model in df['model'].unique():
    df_model = df[df['model'] == model]
    plt.plot(df_model['num_points'], df_model['time_sec'], label=model)
plt.xlabel("Number of Points")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("erwin_runtime_plot.pdf")

# Memory plot
plt.figure()
for model in df['model'].unique():
    df_model = df[df['model'] == model]
    plt.plot(df_model['num_points'], df_model['mem_mb'], label=model)
plt.xlabel("Number of Points")
plt.ylabel("Peak Memory (MB)")
plt.legend()
plt.grid(True)
plt.savefig("erwin_memory_plot.pdf")

# Optional: Compute and print speedups
pivot = df.pivot(index="num_points", columns="model", values="time_sec")
pivot["speedup"] = pivot["ErwinTransformer"] / pivot["ErwinTransformerFlash"]
print(pivot[["speedup"]].mean())