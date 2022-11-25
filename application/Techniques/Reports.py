import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_CF_MSE(lst, k_array, data_info, kind, std_deviation_under, std_deviation_above):

    try:
        #Nome
        extension = ".png"
        fig_name = data_info["name"]
        fig_name += "_MSE"
        fig_name += "_" + str(kind)
        fig_name += extension
        #Caminho
        fig_Path = "Results\\Reports\\"
        fig_Path += fig_name


        sns.set()
        #pal = sns.color_palette("Set2", 6)
        pal = sns.color_palette()
        plt.figure(figsize=(10, 10))
        #Lines
        plt.plot(k_array, lst["user_cos_test_mse"], c=pal[0], linestyle='dashed', label='User-based Cosine', linewidth=5)
        plt.plot(k_array, lst["user_pearson_test_mse"], c=pal[1], linestyle='dashed', label='User-based Pearson', linewidth=5)
        plt.plot(k_array, lst["item_cos_test_mse"], c=pal[2], linestyle='dashed', label='Item-based Cosine', linewidth=5)
        plt.plot(k_array, lst["item_pearson_test_mse"], c=pal[3], linestyle='dashed', label='Item-based Pearson', linewidth=5)
        #Marker

        plt.plot(k_array, lst["user_cos_test_mse"], c=pal[0], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[0], linewidth=5)
        plt.plot(k_array, lst["user_pearson_test_mse"], c=pal[1], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[1], linewidth=5)
        plt.plot(k_array, lst["item_cos_test_mse"], c=pal[2], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[2], linewidth=5)
        plt.plot(k_array, lst["item_pearson_test_mse"], c=pal[3], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[3], linewidth=5)

        if kind == 'average':
            plt.fill_between(k_array, std_deviation_under['user_cos_test_mse'], std_deviation_above['user_cos_test_mse'],  color=pal[0], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['user_pearson_test_mse'], std_deviation_above['user_pearson_test_mse'], color=pal[1], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['item_cos_test_mse'], std_deviation_above['item_cos_test_mse'], color=pal[2], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['item_pearson_test_mse'], std_deviation_above['item_pearson_test_mse'], color=pal[3], alpha=.1)

        plt.legend(loc='best', fontsize=20)
        plt.xticks(fontsize=16);
        plt.yticks(fontsize=16);
        plt.xlabel('k', fontsize=30);
        plt.ylabel('MSE', fontsize=30);
        plt.savefig(fig_Path)

    except Exception as error:
        print(error)

def plot_CF_MAE(lst, k_array, data_info, kind, std_deviation_under, std_deviation_above):

    try:
        #Nome
        extension = ".png"
        fig_name = data_info["name"]
        fig_name += "_MAE"
        fig_name += "_" + str(kind)
        fig_name += extension
        #Caminho
        fig_Path = "Results\\Reports\\"
        fig_Path += fig_name


        sns.set()
        #pal = sns.color_palette("Set2", 6)
        pal = sns.color_palette()
        plt.figure(figsize=(10, 10))
        #Lines
        plt.plot(k_array, lst["user_cos_test_mae"], c=pal[0], linestyle='dashed', label='User-based Cosine', linewidth=5)
        plt.plot(k_array, lst["user_pearson_test_mae"], c=pal[1], linestyle='dashed', label='User-based Pearson', linewidth=5)
        plt.plot(k_array, lst["item_cos_test_mae"], c=pal[2], linestyle='dashed', label='Item-based Cosine', linewidth=5)
        plt.plot(k_array, lst["item_pearson_test_mae"], c=pal[3], linestyle='dashed', label='Item-based Pearson', linewidth=5)
        #Marker

        plt.plot(k_array, lst["user_cos_test_mae"], c=pal[0], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[0], linewidth=5)
        plt.plot(k_array, lst["user_pearson_test_mae"], c=pal[1], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[1], linewidth=5)
        plt.plot(k_array, lst["item_cos_test_mae"], c=pal[2], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[2], linewidth=5)
        plt.plot(k_array, lst["item_pearson_test_mae"], c=pal[3], linestyle='None', marker='o', markersize=12, markerfacecolor=pal[3], linewidth=5)

        if kind == 'average':
            plt.fill_between(k_array, std_deviation_under['user_cos_test_mae'], std_deviation_above['user_cos_test_mae'],  color=pal[0], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['user_pearson_test_mae'], std_deviation_above['user_pearson_test_mae'], color=pal[1], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['item_cos_test_mae'], std_deviation_above['item_cos_test_mae'], color=pal[2], alpha=.1)
            plt.fill_between(k_array, std_deviation_under['item_pearson_test_mae'], std_deviation_above['item_pearson_test_mae'], color=pal[3], alpha=.1)

        plt.legend(loc='best', fontsize=20)
        plt.xticks(fontsize=16);
        plt.yticks(fontsize=16);
        plt.xlabel('k', fontsize=30);
        plt.ylabel('MAE', fontsize=30);
        plt.savefig(fig_Path)

    except Exception as error:
        print(error)


def plot_data_info():
    # Nome
    extension = ".png"
    fig_name = 'Dados_Caracterizacao'
    # Caminho
    fig_Path = "Results\\Reports\\"
    fig_Path += fig_name

    labels = ['MovieLens', 'Amazon', 'Book-Crossing']
    users = [943, 17788, 6400]
    items = [1682, 3449, 8698]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, users, width, label='Usuários')
    rects2 = ax.bar(x + width / 2, items, width, label='Itens')

    # Labels e títulos
    ax.set_ylabel('Total')
    ax.set_title('Total de usuários e itens por conjunto de dados')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    plt.savefig(fig_Path)

def plot_info_rating():
    # Nome
    extension = ".png"
    fig_name = 'Dados_Ratings'
    # Caminho
    fig_Path = "Results\\Reports\\"
    fig_Path += fig_name

    ratings = [100000, 20984, 10000]

    labels = ['MovieLens', 'Amazon', 'Book-Crossing']
    users = [ratings[0]/943, ratings[1]/17788, ratings[2]/6400]
    items = [ratings[0]/1682, ratings[1]/3449, ratings[2]/8698]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, users, width, label='Usuários')
    rects2 = ax.bar(x + width / 2, items, width, label='Itens')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Média de avaliações')
    ax.set_title('Quantidade média de avaliações por conjunto de dados')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    plt.savefig(fig_Path)





