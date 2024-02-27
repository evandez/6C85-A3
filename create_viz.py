import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("census2000.csv")


# Step 1: Preprocess the dataset.
grouped_df = df.groupby(["Year", "Age"]).agg({"People": "sum"}).reset_index()

# Turn People into a percentage of the total population for each year.
total_population_by_year = grouped_df.groupby("Year")["People"].transform("sum")
grouped_df["People"] = grouped_df["People"] / total_population_by_year * 100


def flatten(x):
    return np.array([x[-1] for x in x])


pop_1900 = flatten(
    grouped_df[grouped_df["Year"] == 1900][["Age", "People"]].values.tolist()
)
pop_2000 = flatten(
    grouped_df[grouped_df["Year"] == 2000][["Age", "People"]].values.tolist()
)

age_groups = [f"{i}-{j}" for i, j in zip(list(range(0, 95, 5)), list(range(4, 95, 5)))]
assert len(age_groups) == len(pop_1900) == len(pop_2000)


# Compute median ages for later.
def get_median_index(population):
    cumsum = 0
    for i, p in enumerate(population):
        cumsum += p
        if cumsum >= 50:
            return i
    return len(population) - 1


med_1900 = get_median_index(pop_1900)
med_2000 = get_median_index(pop_2000)

# Plot the thing.
fig, ax = plt.subplots(figsize=(10, 8))

# Create horizontal bars; use negative values for 1900 so that bars
# appear on the left side, to give a "before" vs "after" comparison.
for i in range(len(age_groups)):
    ax.barh(
        age_groups[i],
        -pop_1900[i],
        color="#E97451",
        hatch="//" if i == med_1900 else None,
    )
    ax.barh(
        age_groups[i],
        pop_2000[i],
        color="#008080",
        hatch="//" if i == med_2000 else None,
    )
ax.plot([0, 0], [-1, 19], color="black", lw=1)

# Adding labels and title
ax.set_xlabel("Percentage of Total Population", fontsize=16)
ax.set_ylabel("Age Group", fontsize=16)
ax.set_title("The Changing Shape of the U.S. Population", fontsize=24, pad=20)
ax.set_yticks(range(len(age_groups)))  # prevents warning
ax.set_yticklabels(age_groups)
ax.set_xticks([-13, -6.5, 0, 6.5, 13])
ax.set_xticklabels(["13%", "6.5%", "0", "6.5%", "13%"])

ax.text(-8, 16, "1900", va="center", ha="center", backgroundcolor="w", fontsize=24)
ax.text(8, 16, "2000", va="center", ha="center", backgroundcolor="w", fontsize=24)

# Add a hacky legend.
ax.text(
    9.35,
    12,
    "Median\n",
    va="center",
    ha="center",
    fontsize=12,
    bbox=dict(edgecolor="black", facecolor="none", alpha=0.5, pad=10),
)
example_rect = plt.Rectangle(
    (8.35, 11.25),
    2,
    0.5,
    linewidth=1,
    edgecolor="black",
    facecolor="none",
    hatch="//",
)
ax.add_patch(example_rect)

# Show plot
plt.savefig("figures/final_viz.png")
