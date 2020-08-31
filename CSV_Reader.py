class CSV_Reader():
    def getPrices(filename,enc):
        csv_file = open(filename, "r", encoding=enc, errors="", newline="" )
        f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        loop = 2
        
        for itr in range(loop):
            next(f)
        latest_price_list = []
        for row in f:
            latest_price_list.append(float(row[4].replace(',', '')))
        latest_price_list = np.array(latest_price_list)
        return latest_price_list



