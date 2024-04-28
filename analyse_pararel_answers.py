import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def calculate_matching_percentage(lst):
    count = 0
    total_pairs = 0

    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            total_pairs += 1
            if lst[i] == lst[j]:
                count += 1

    if total_pairs == 0:
        return 0
    else:
        return (count / total_pairs) * 100


def get_row_wise_consistency(results, model):
    ## consistency as defined in the paralel paper

    consistency_list = []

    for idx, row in results.iterrows():
        predicted_answers = row[f"{model}_answers"]

        consistency = calculate_matching_percentage(predicted_answers)
        consistency_list.append(consistency)

    return np.mean(consistency_list)


def get_row_wise_acc(results, model):
    acc_list = []
    for idx, row in results.iterrows():
        true_answer = row["true_answer"]
        predicted_answers = row[f"{model}_answers"]
        # for key, value in row["logprobs"].items():
        # 	predicted_answers.append(value[-1])

        correct_answers = [
            predicted_answer == true_answer for predicted_answer in predicted_answers
        ]
        acc = correct_answers.count(1) / len(correct_answers) * 100
        acc_list.append(acc)

    return np.mean(acc_list)


def plot_result(acc, consistency, models):
    # Create the plot
    plt.figure(figsize=(15, 9))
    plt.scatter(acc, consistency)

    # Label each point
    for i, txt in enumerate(models):
        plt.annotate(txt, (acc[i], consistency[i]))

    # Label axes and set title
    plt.xlabel("Accuracy (acc)")
    plt.ylabel("Consistency")
    plt.title("Model Consistency vs Accuracy")

    # Show the plot
    plt.show()


def plot_all(models, results):
    # Create the plot
    plt.figure(figsize=(17.5, 10.5))

    # set up shapes and colours for all models
    markers = ["o", "v", "s", "P", "^", "d", "X", "*"]
    marker_labels = [
        "Zero-shot",
        "2-shot",
        "4-shot",
        "6-shot",
        "CoT (2-shot)",
        "CoT (4-shot)",
        "CoT (6-shot)",
        "Self-Consistency (CoT + 2-shot)",
    ]
    sizes = [150.0, 120.0, 120.0, 170.0, 120.0, 120.0, 130.0, 190.0]
    colors = {
        "ada": "cornflowerblue",
        "text-ada-001": "royalblue",
        "babbage": "mediumorchid",
        "text-babbage-001": "darkorchid",
        "curie": "salmon",
        "text-curie-001": "crimson",
        "davinci": "yellow",
        "text-davinci-001": "gold",
        "text-davinci-002": "orange",
        "text-davinci-003": "darkgoldenrod",
        "gpt-3.5-turbo-instruct": "green",
        "gpt-3.5-turbo": "limegreen",
        "gpt-4": "turquoise",
    }

    # zip together the various results to get a list containing results for each model
    accs = list(results.values())[::2]
    cons = list(results.values())[1::2]
    accs = list(zip(*accs))
    cons = list(zip(*cons))

    # create empty legends
    color_legend_lines = []
    marker_legend_lines = []

    # plot each model's results
    for i, model_name in enumerate(models):
        color = colors[model_name]
        for j in range(len(accs[i])):
            plt.scatter(
                accs[i][j],
                cons[i][j],
                s=sizes[j],
                marker=markers[j],
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=3,
            )
        line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=model_name,
        )
        color_legend_lines.append(line)

        # draw a line of best fit for this model's points
        a, b = np.polyfit(accs[i], cons[i], 1)
        plt.plot(
            accs[i], [a * x + b for x in accs[i]], color=color, linewidth=4.3, zorder=2
        )

        # add marker legend lines for first model only
        if i == 0:
            for m, label in zip(markers, marker_labels):
                line = plt.Line2D(
                    [0],
                    [0],
                    marker=m,
                    color="white",
                    markerfacecolor="black",
                    markersize=10,
                    label=label,
                )
                marker_legend_lines.append(line)

    # add a grid
    plt.grid(
        which="both",
        axis="both",
        color="black",
        alpha=0.1,
        linewidth=1,
        linestyle="--",
    )
    plt.grid(
        which="major",
        axis="both",
        color="black",
        alpha=0.2,
        linewidth=1.5,
        linestyle="-",
    )

    # Label axes and set title
    ax1 = plt.gca()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(labelsize=12)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Accuracy", fontsize=14)
    ax1.set_ylabel("Consistency", fontsize=14)

    plt.title("Model Performances on ParaRel Dataset", fontsize=18, pad=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(labelsize=12)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Consistency", fontsize=14)

    # add legends
    color_legend_lines = color_legend_lines[::2] + color_legend_lines[1::2]
    legend1 = plt.legend(
        handles=color_legend_lines, loc="upper left", ncols=2, prop={"size": 14}
    )
    legend2 = plt.legend(
        handles=marker_legend_lines, loc="lower right", prop={"size": 12}
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Show the plot
    plt.show()

def plot_all_simplified_old(models, results):
    # Create the plot
    plt.figure(figsize=(17.5, 10.5))

    # set up shapes and colours for all models
    # markers = ["o", "^", "s", "*"]
    markers = ["o", "v", "s", "P", "^", "d", "X", "*"]
    # marker_labels = [
    #     "Zero-shot",
    #     "Few-shot",
    #     "Chain of Thought",
    #     "Self-Consistency",
    # ]
    marker_labels = [
        "Zero-shot",
        "2-shot",
        "4-shot",
        "6-shot",
        "CoT (2-shot)",
        "CoT (4-shot)",
        "CoT (6-shot)",
        "Self-Consistency (CoT + 2-shot)",
    ]
    # sizes = [150.0, 120.0, 120.0, 190.0]
    sizes = [150.0, 120.0, 120.0, 170.0, 120.0, 120.0, 130.0, 190.0]
    colors = {
        "ada": "cornflowerblue",
        "text-ada-001": "royalblue",
        "babbage": "mediumorchid",
        "text-babbage-001": "darkorchid",
        "curie": "salmon",
        "text-curie-001": "crimson",
        "davinci": "yellow",
        "text-davinci-001": "gold",
        "text-davinci-002": "orange",
        "text-davinci-003": "darkgoldenrod",
        "gpt-3.5-turbo-instruct": "green",
        "gpt-3.5-turbo": "limegreen",
        "gpt-4": "turquoise",
    }

    # zip together the various results to get a list containing results for
    # each model; take the mean of few-shot and CoT results
    accs = list(results.values())[::2]
    cons = list(results.values())[1::2]
    # accs = [accs[0], map(np.mean, zip(accs[1], accs[2], accs[3])), map(np.mean, zip(accs[4], accs[5], accs[6])), accs[7]]
    # cons = [cons[0], map(np.mean, zip(cons[1], cons[2], cons[3])), map(np.mean, zip(cons[4], cons[5], cons[6])), cons[7]]
    accs = list(zip(*accs))
    cons = list(zip(*cons))

    # create empty legends
    color_legend_lines = []
    marker_legend_lines = []

    # plot each model's results
    for i, model_name in enumerate(models):
        color = colors[model_name]
        for j in range(len(accs[i])):
            plt.scatter(
                accs[i][j],
                cons[i][j],
                s=sizes[j],
                marker=markers[j],
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=5 if j==0 else 3,
                alpha=1 if j==0 else 0.25,
            )
        line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=model_name,
        )
        color_legend_lines.append(line)

        # calculate the mean and std. deviation of all other points
        mean_acc = np.mean(accs[i][1:]) # excluding the first point
        mean_con = np.mean(cons[i][1:]) # excluding the first point
        std_acc = np.std(accs[i][1:]) # standard deviation
        std_con = np.std(cons[i][1:]) # standard deviation

        # create the upper and lower bounds of the cone
        upper_acc = mean_acc + std_acc
        upper_con = mean_con + std_con
        lower_acc = mean_acc - std_acc
        lower_con = mean_con - std_con

        # create a gradient effect for the shaded area
        num_segments = 100  # increase the number of segments for a smoother gradient
        acc_segments = np.linspace(accs[i][0], mean_acc, num_segments)
        con_lower_segments = np.linspace(cons[i][0], lower_con, num_segments)
        con_upper_segments = np.linspace(cons[i][0], upper_con, num_segments)
        distances = np.sqrt((acc_segments-accs[i][0])**2 + (0.5*(con_lower_segments+con_upper_segments)-cons[i][0])**2)  # calculate the distance of each segment from the starting point
        max_distance = np.max(distances)
        alpha_values = 0.3 * (1 - distances / max_distance)  # calculate the alpha value for each segment based on its distance from the starting point
        for k in range(num_segments-1):
            plt.fill_between(acc_segments[k:k+2], con_lower_segments[k:k+2], con_upper_segments[k:k+2], color=color, alpha=alpha_values[k], zorder=4)

        # draw an arrow from the first point to the mean of all other points with a black border
        plt.arrow(accs[i][0], cons[i][0], mean_acc-accs[i][0], mean_con-cons[i][0], width=0.5, head_width=1.25, head_length=0.8, fc="white", ec="white", lw=0.8, zorder=4, alpha=1)
        plt.arrow(accs[i][0], cons[i][0], mean_acc-accs[i][0], mean_con-cons[i][0], width=0.5, head_width=1.25, head_length=0.8, fc=color, ec="black", lw=0.8, zorder=4, alpha=0.75)

        # add marker legend lines for first model only
        if i == 0:
            for m, label in zip(markers, marker_labels):
                line = plt.Line2D(
                    [0],
                    [0],
                    marker=m,
                    color="white",
                    markerfacecolor="black",
                    markersize=10,
                    label=label,
                )
                marker_legend_lines.append(line)

    # add a grid
    plt.grid(
        which="both",
        axis="both",
        color="black",
        alpha=0.1,
        linewidth=1,
        linestyle="--",
    )
    plt.grid(
        which="major",
        axis="both",
        color="black",
        alpha=0.2,
        linewidth=1.5,
        linestyle="-",
    )

    # Label axes and set title
    ax1 = plt.gca()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(labelsize=12)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Accuracy", fontsize=14)
    ax1.set_ylabel("Consistency", fontsize=14)

    plt.title("Model Performances on ParaRel Dataset", fontsize=18, pad=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(labelsize=12)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Consistency", fontsize=14)

    # add legends
    color_legend_lines = color_legend_lines[::2] + color_legend_lines[1::2]
    legend1 = plt.legend(
        handles=color_legend_lines, loc="upper left", ncols=2, prop={"size": 14}
    )
    legend2 = plt.legend(
        handles=marker_legend_lines, loc="lower right", prop={"size": 12}
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Show the plot
    plt.show()

# creates a simplified and cleaned up plot of all experimental results
def plot_all_simplified(models, results):
    # Create the plot
    plt.figure(figsize=(17.5, 10.5))

    small_models = [
        "ada", "text-ada-001", "babbage", "text-babbage-001", "curie", "text-curie-001",
    ]

    # set up shapes and colours for all models
    # markers = ["o", "^", "s", "*"]
    markers = ["o", "v", "s", "P", "^", "d", "X", "*"]
    # marker_labels = [
    #     "Zero-shot",
    #     "Few-shot",
    #     "Chain of Thought",
    #     "Self-Consistency",
    # ]
    marker_labels = [
        "Zero-shot",
        "2-shot",
        "4-shot",
        "6-shot",
        "CoT (2-shot)",
        "CoT (4-shot)",
        "CoT (6-shot)",
        "Self-Consistency (CoT + 2-shot)",
    ]
    # sizes = [150.0, 120.0, 120.0, 190.0]
    sizes = [150.0, 120.0, 120.0, 170.0, 120.0, 120.0, 130.0, 190.0]
    colors = {
        "ada": "cornflowerblue",
        "text-ada-001": "royalblue",
        "babbage": "mediumorchid",
        "text-babbage-001": "darkorchid",
        "curie": "salmon",
        "text-curie-001": "crimson",
        "davinci": "yellow",
        "text-davinci-001": "gold",
        "text-davinci-002": "orange",
        "text-davinci-003": "darkgoldenrod",
        "gpt-3.5-turbo-instruct": "green",
        "gpt-3.5-turbo": "limegreen",
        "gpt-4": "turquoise",
    }

    # zip together the various results to get a list containing results for
    # each model; take the mean of few-shot and CoT results
    small_accs = results["zero_accs"][:6]
    small_cons = results["zero_cons"][:6]
    results["zero_accs"] = results["zero_accs"][6:]
    results["zero_cons"] = results["zero_cons"][6:]
    accs = list(results.values())[::2]
    cons = list(results.values())[1::2]
    # accs = [accs[0], map(np.mean, zip(accs[1], accs[2], accs[3])), map(np.mean, zip(accs[4], accs[5], accs[6])), accs[7]]
    # cons = [cons[0], map(np.mean, zip(cons[1], cons[2], cons[3])), map(np.mean, zip(cons[4], cons[5], cons[6])), cons[7]]
    accs = accs[:-2]
    cons = cons[:-2]
    accs = list(zip(*accs))
    cons = list(zip(*cons))

    # create empty legends
    color_legend_lines = []
    marker_legend_lines = []

    # plot each model's results
    for i, model_name in enumerate(models):
        if model_name in small_models:
            color = colors[model_name]
            plt.scatter(
                small_accs[small_models.index(model_name)],
                small_cons[small_models.index(model_name)],
                s=sizes[0],
                marker=markers[0],
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=5,
                alpha=1,
            )
            line = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
                label=model_name,
            )
            color_legend_lines.append(line)
            
        else:
            color = colors[model_name]
            for j in range(len(accs[i-6])):
                plt.scatter(
                    accs[i-6][j],
                    cons[i-6][j],
                    s=sizes[j],
                    marker=markers[j],
                    color=color,
                    edgecolors="black",
                    linewidths=1,
                    zorder=5 if j == 0 else 3,
                    alpha=1 if j == 0 else 0.25,
                )
            line = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
                label=model_name,
            )
            color_legend_lines.append(line)

            # calculate the mean and std. deviation of all other points
            mean_acc = np.mean(accs[i-6][1:])  # excluding the first point
            mean_con = np.mean(cons[i-6][1:])  # excluding the first point
            std_acc = np.std(accs[i-6][1:])  # standard deviation
            std_con = np.std(cons[i-6][1:])  # standard deviation

            # # create the upper and lower bounds of the cone
            # upper_acc = mean_acc + std_acc
            # upper_con = mean_con + std_con
            # lower_acc = mean_acc - std_acc
            # lower_con = mean_con - std_con

            # # create a gradient effect for the shaded area
            # num_segments = 100  # increase the number of segments for a smoother gradient
            # acc_segments = np.linspace(accs[i][0], mean_acc, num_segments)
            # con_lower_segments = np.linspace(cons[i][0], lower_con, num_segments)
            # con_upper_segments = np.linspace(cons[i][0], upper_con, num_segments)
            # distances = np.sqrt(
            #     (acc_segments - accs[i][0]) ** 2
            #     + (0.5 * (con_lower_segments + con_upper_segments) - cons[i][0]) ** 2
            # )  # calculate the distance of each segment from the starting point
            # max_distance = np.max(distances)
            # alpha_values = 0.3 * (
            #     1 - distances / max_distance
            # )  # calculate the alpha value for each segment based on its distance from the starting point
            # for k in range(num_segments - 1):
            #     plt.fill_between(
            #         acc_segments[k : k + 2],
            #         con_lower_segments[k : k + 2],
            #         con_upper_segments[k : k + 2],
            #         color=color,
            #         alpha=alpha_values[k],
            #         zorder=4,
            #     )

            # draw an arrow from the first point to the mean of all other points with a black border
            plt.arrow(
                accs[i-6][0],
                cons[i-6][0],
                mean_acc - accs[i-6][0],
                mean_con - cons[i-6][0],
                width=0.3,
                head_width=0.8,
                head_length=0.5,
                fc="white",
                ec="white",
                lw=0.6,
                zorder=4,
                alpha=1,
            )
            plt.arrow(
                accs[i-6][0],
                cons[i-6][0],
                mean_acc - accs[i-6][0],
                mean_con - cons[i-6][0],
                width=0.3,
                head_width=0.8,
                head_length=0.5,
                fc=color,
                ec="black",
                lw=0.6,
                zorder=4,
                alpha=0.75,
            )

            # add marker legend lines for first model only
            if i == 6:
                for m, label in zip(markers, marker_labels):
                    line = plt.Line2D(
                        [0],
                        [0],
                        marker=m,
                        color="white",
                        markerfacecolor="black",
                        markersize=10,
                        label=label,
                    )
                    marker_legend_lines.append(line)

    # add a grid
    plt.grid(
        which="both",
        axis="both",
        color="black",
        alpha=0.1,
        linewidth=1,
        linestyle="--",
    )
    plt.grid(
        which="major",
        axis="both",
        color="black",
        alpha=0.2,
        linewidth=1.5,
        linestyle="-",
    )

    # Label axes and set title
    ax1 = plt.gca()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(labelsize=12)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Accuracy", fontsize=14)
    ax1.set_ylabel("Consistency", fontsize=14)

    plt.title("Model Performances on Scenarios Dataset", fontsize=18, pad=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(labelsize=12)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Consistency", fontsize=14)

    # add legends
    color_legend_lines = color_legend_lines[::2] + color_legend_lines[1::2]
    legend1 = plt.legend(
        handles=color_legend_lines, loc="upper left", ncols=2, prop={"size": 14}
    )
    legend2 = plt.legend(
        handles=marker_legend_lines, loc="lower right", prop={"size": 12}
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Show the plot
    plt.show()

if __name__ == "__main__":
	# the list of models to be plotted on the graph
	models = [
		"ada",
		"text-ada-001",
		"babbage",
		"text-babbage-001",
		"curie",
		"text-curie-001",
		"davinci",
		"text-davinci-001",
		"text-davinci-002",
		"text-davinci-003",
		"gpt-3.5-turbo-instruct",
		"gpt-3.5-turbo",
		"gpt-4",
        # "gpt-3.5-turbo_0",
        # "gpt-3.5-turbo_25",
        # "gpt-3.5-turbo_50",
        # "gpt-3.5-turbo_75",
        # "gpt-3.5-turbo_100",
        # 'davinci-002_conv-0',
        # 'davinci-002_conv-25',
        # 'davinci-002_conv-50',
        # 'davinci-002_75',
        # 'davinci-002_100',
	]

	small_models = [
        "ada",
        "text-ada-001",
        "babbage",
        "text-babbage-001",
        "curie",
        "text-curie-001",
    ]

	# num_question is used to make sure the correct data is plotted
	#   = how many questions were asked to the models
	num_questions = 125

	# set up the results dict to contain all the results
	results_dict = {}
	results_dict["zero_accs"] = []
	results_dict["zero_cons"] = []
	results_dict["two_accs"] = []
	results_dict["two_cons"] = []
	results_dict["four_accs"] = []
	results_dict["four_cons"] = []
	results_dict["six_accs"] = []
	results_dict["six_cons"] = []
	results_dict["cot_two_accs"] = []
	results_dict["cot_two_cons"] = []
	results_dict["cot_four_accs"] = []
	results_dict["cot_four_cons"] = []
	results_dict["cot_six_accs"] = []
	results_dict["cot_six_cons"] = []
	results_dict["scs_cot_two_accs"] = []
	results_dict["scs_cot_two_cons"] = []

	# each of the possible settings for the experiments (comment out unwanted
	# lines to get results for a particular set of experiments)
	settings = [
		{
			"few_shot": False,
			"num_shots": 0,
			"cot": False,
			"scs": False,
			"exp_size": num_questions,
		},  # zero-shot
		{
			"few_shot": True,
			"num_shots": 2,
			"cot": False,
			"scs": False,
			"exp_size": num_questions,
		},  # 2-shot
		{
			"few_shot": True,
			"num_shots": 4,
			"cot": False,
			"scs": False,
			"exp_size": num_questions,
		},  # 4-shot
		{
			"few_shot": True,
			"num_shots": 6,
			"cot": False,
			"scs": False,
			"exp_size": num_questions,
		},  # 6-shot
		{
			"few_shot": True,
			"num_shots": 2,
			"cot": True,
			"scs": False,
			"exp_size": num_questions,
		},  # CoT 2S
		{
			"few_shot": True,
			"num_shots": 4,
			"cot": True,
			"scs": False,
			"exp_size": num_questions,
		},  # CoT 4S
		{
			"few_shot": True,
			"num_shots": 6,
			"cot": True,
			"scs": False,
			"exp_size": num_questions,
		},  # CoT 6S
		{
			"few_shot": True,
			"num_shots": 2,
			"cot": True,
			"scs": True,
			"exp_size": num_questions,
		},  # SCS CoT 2S
	]

	# iterate over all the settings and store the model's results in the dict
	for setting in settings:
		exp_size = setting["exp_size"]
		
		# zero-shot setup
		if not setting["few_shot"]:
			results = pd.read_json(f"pararel_answers_{exp_size}.jsonl", lines=True)
			small_results = pd.read_json(f"pararel_answers_10.jsonl", lines=True)
			entry = "zero"
		
		# few-shot setup
		elif not setting["cot"]:
			results = pd.read_json(
				f"pararel_answers_{setting['num_shots']}S_{exp_size}.jsonl", lines=True
			)
			if setting["num_shots"] == 2:
				entry = "two"
			elif setting["num_shots"] == 4:
				entry = "four"
			else:
				entry = "six"
		
		# CoT setup - get FS results too for replacing garbage
		elif not setting["scs"]:
			results = pd.read_json(
				f"pararel_answers_CoT_{setting['num_shots']}S_{exp_size}.jsonl",
				lines=True,
			)
			if setting["num_shots"] == 2:
				entry = "cot_two"
			elif setting["num_shots"] == 4:
				entry = "cot_four"
			else:
				entry = "cot_six"
			FS_results = pd.read_json(
				f"pararel_answers_{setting['num_shots']}S_{exp_size}.jsonl", lines=True
			)
		
		# SCS setup - get FS results too for replacing garbage
		else:
			results = pd.read_json(
				f"pararel_answers_SCS_CoT_{setting['num_shots']}S_{exp_size}.jsonl",
				lines=True,
			)
			if setting["num_shots"] == 2:
				entry = "scs_cot_two"
			elif setting["num_shots"] == 4:
				entry = "scs_cot_four"
			else:
				entry = "scs_cot_six"
			FS_results = pd.read_json(
				f"pararel_answers_{setting['num_shots']}S_{exp_size}.jsonl", lines=True
			)
		
		# set up where accs and cons analysis will be put in results_dict
		accs_name = entry + "_accs"
		cons_name = entry + "_cons"

		# iterate over each model and analyse its answers
		for model in models:

			if (model in small_models) and (setting["few_shot"]):
				continue
			# if looking at CoT or SCS answers, replace any garbage answers
			if setting["cot"]:
				# iterate over this model's answers in the dataframe
				for idx, row in results.iterrows():
					for i, ans in enumerate(row[f"{model}_answers"]):
						if ans == "garbage":
							# find the corresponding few-shot answer and replace
							few_shot_ans = FS_results.iloc[idx][f"{model}_answers"][i]
							results.iloc[idx][f"{model}_answers"][i] = few_shot_ans
			if model in small_models:
				results = small_results
			acc = get_row_wise_acc(results, model)
			consistency = get_row_wise_consistency(results, model)
			results_dict[accs_name].append(acc)
			results_dict[cons_name].append(consistency)

    # run the `plot_all` method
	plot_all_simplified(models, results_dict)
