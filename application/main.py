import os
import sys
from Common import Service
from Techniques import CollaborativeFiltering
import Utils
from Techniques import Reports


def builDataPath(id):
    dataInfo = {

    }
    thisDir = os.path.dirname(__file__)

    if id == 0:

        dataPath_0 = "Dataset\\ml-100k"
        dataFile_0 = "u.data"
        finalPath_0 = ""
        finalPath_0 = thisDir + "\\" + dataPath_0 + "\\" + dataFile_0

        dataInfo["id"] = id
        dataInfo["path"] = finalPath_0
        dataInfo["name"] = "MovieLens"

        return dataInfo

    elif id == 1:

        dataPath_1 = "Dataset\\Amazon"
        dataFile_1 = "ratings_Amazon_Instant_Video.csv"

        finalPath_0 = ""
        finalPath_0 = thisDir + "\\" + dataPath_1 + "\\" + dataFile_1

        dataInfo["id"] = id
        dataInfo["path"] = finalPath_0
        dataInfo["name"] = "Amazon_Video"

        return dataInfo

    elif id == 2:

        dataPath_1 = "Dataset\\Book_Ratings"
        dataFile_1 = "BX-Book-Ratings.csv"

        finalPath_0 = ""
        finalPath_0 = thisDir + "\\" + dataPath_1 + "\\" + dataFile_1

        dataInfo["id"] = id
        dataInfo["path"] = finalPath_0
        dataInfo["name"] = "Book_Ratings"

        return dataInfo

    return dataInfo


def start():
    idMovieLens = 0
    idAmazon_music = 1
    idBook_rating = 2

    dict_data_info = {}
    dict_data_info[idMovieLens] = builDataPath(0)
    dict_data_info[idAmazon_music] = builDataPath(1)
    dict_data_info[idBook_rating] = builDataPath(2)

    # Plot informacoes dos dados
    Reports.plot_data_info()
    Reports.plot_info_rating()

    try:
        # TO DO mudar para 0
        for key, data_info in dict_data_info.items():

            if data_info["id"] == 0:
                movieLensDB = Service.SearchData(data_info)
                Utils.analyze(movieLensDB, data_info)
                print(data_info)
                movieLensDB.hist()
                CollaborativeFiltering.run(movieLensDB, data_info)

            if data_info["id"] == 1:
                amazonDB = Service.SearchData(data_info)
                Utils.analyze(amazonDB, data_info)
                print(data_info)
                amazonDB.hist()
                CollaborativeFiltering.run(amazonDB, data_info)

            if data_info["id"] == 2:
                bookDB = Service.SearchData(data_info)
                Utils.analyze(bookDB, data_info)
                print(data_info)
                bookDB.hist()
                CollaborativeFiltering.run(bookDB, data_info)

    except Exception as error:
        print(error)


if __name__ == "__main__":
    start()
