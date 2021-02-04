import openpyxl


class legacyWriter:
    path = ""
    sheet = ""

    def __init__(self, path, sheet):
        self.path = path
        self.sheet = sheet
        self.wb = openpyxl.load_workbook(path)  # load the workbook"
        self.worksheet = self.wb.get_sheet_by_name(sheet)


    def write_uniformity_body_report(self, df):
        print(f"[INFO] writing uniformity mode [BODY] test results on {self.path} sheet: {self.sheet} ")
        #start changing values

        #ggiornato template

        #mean values
        self.worksheet["C26"]=df["water_ct"].values[0]
        self.worksheet["C27"] = df["uniformity_up"].values[0]
        self.worksheet["C28"] = df["uniformity_bottom"].values[0]
        self.worksheet["C29"] = df["uniformity_right"].values[0]
        self.worksheet["C30"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D26"]=df["noise"].values[0]
        self.worksheet["D27"] = df["std_up"].values[0]
        self.worksheet["D28"] = df["std_bottom"].values[0]
        self.worksheet["D29"] = df["std_right"].values[0]
        self.worksheet["D30"] = df["std_left"].values[0]

        self.wb.save(self.path)


    def write_uniformity_head_report(self, df):
        print(f"[INFO] writing uniformity mode [HEAD] test results on {self.path} sheet: {self.sheet} ")
        #ggiornato template

        #mean values
        self.worksheet["C12"]=df["water_ct"].values[0]
        self.worksheet["C13"] = df["uniformity_up"].values[0]
        self.worksheet["C14"] = df["uniformity_bottom"].values[0]
        self.worksheet["C15"] = df["uniformity_right"].values[0]
        self.worksheet["C16"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D12"]=df["noise"].values[0]
        self.worksheet["D13"] = df["std_up"].values[0]
        self.worksheet["D14"] = df["std_bottom"].values[0]
        self.worksheet["D15"] = df["std_right"].values[0]
        self.worksheet["D16"] = df["std_left"].values[0]

        self.wb.save(self.path)

    def write_uniformity_monoenergetic_report(self,df):
        print(f"[INFO] writing uniformity mode [monoenergetic] test results on {self.path} sheet: {self.sheet} ")
        #start changing values
        #aggiornato template

        #mean values
        self.worksheet["C126"]=df["water_ct"].values[0]
        self.worksheet["C127"] = df["uniformity_up"].values[0]
        self.worksheet["C128"] = df["uniformity_bottom"].values[0]
        self.worksheet["C129"] = df["uniformity_right"].values[0]
        self.worksheet["C130"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D126"]=df["noise"].values[0]
        self.worksheet["D127"] = df["std_up"].values[0]
        self.worksheet["D128"] = df["std_bottom"].values[0]
        self.worksheet["D129"] = df["std_right"].values[0]
        self.worksheet["D130"] = df["std_left"].values[0]

        self.wb.save(self.path)



    def write_uniformity_multislice_report(self, df,option):
        print(f"[INFO] writing uniformity mode [MULTISLICE] test results on {self.path} sheet: {self.sheet} ")
        #start changing values
        #aggiornato
        #mean values
        if option=="multislice":
            start_row=43
        elif option=="multislice_monoenergetic":
            start_row=142
        for i in range(len(df)):
            row=start_row+i
            self.worksheet[f"B{row}"]=df.loc[i,"water_ct"]
            self.worksheet[f"C{row}"] = df.loc[i,"uniformity_up"]
            self.worksheet[f"D{row}"] = df.loc[i,"uniformity_bottom"]
            self.worksheet[f"E{row}"] = df.loc[i,"uniformity_right"]
            self.worksheet[f"F{row}"] = df.loc[i,"uniformity_left"]



        self.wb.save(self.path)

    def write_linearity_report(self, lin,option):

        #aggiornato con il nuovo template

        if option=="fbp":
            letter="C"
        elif option=="asir":
            letter="G"

        print(f"[INFO] writing linearity  test results on {self.path} sheet: {self.sheet} ")
        for i,val in enumerate(lin):
            self.worksheet[f"{letter}{8+i}"] = val

        self.wb.save(self.path)

    def write_resolution_report(self, means, w, p, stds, std_w, std_p, filter):
        #aggiornato al nuovo template

        print(f"[INFO] writing resolution  test results on {self.path} sheet: {self.sheet} ")

        if filter=="std":
            start = 72
            self.worksheet[f"B67"] = p
            self.worksheet[f"B68"] = w

            self.worksheet[f"B77"] = w
            self.worksheet[f"C77"] = std_w

            self.worksheet[f"B78"] = p
            self.worksheet[f"C78"] = std_p

        elif filter=="bone":
            start=88
            self.worksheet[f"B83"] = p
            self.worksheet[f"B84"] = w

            self.worksheet[f"B93"] = w
            self.worksheet[f"C93"] = std_w

            self.worksheet[f"B94"] = p
            self.worksheet[f"C94"] = std_p



        for i, val in enumerate(means):
            self.worksheet[f"B{start + i}"] = means[i]
            self.worksheet[f"C{start + i}"] = stds[i]
        #self.worksheet[f"B{start + 5}"] = w
        #self.worksheet[f"B{start + 6}"] = p
        #self.worksheet[f"C{start + 5}"] = std_w
        #self.worksheet[f"C{start + 6}"] = std_p



        self.wb.save(self.path)

    def write_dot_thickness_report(self, fwhm):

        #aggiornato al nuovo template
        print(f"[INFO] writing linearity  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["C22"] = fwhm

        self.wb.save(self.path)

    def write_lowcontrast_report(self, mu_list,std_list, lcd):

        #aggiornato al nuovo template

        print(f"[INFO] writing lowcontrast  test results on {self.path} sheet: {self.sheet} ")

        #self.worksheet["A112"] = mu_list[0]
        #self.worksheet["B112"] = std_list[0]
        #self.worksheet["C112"] = lcd[0]

        start=103
        for i in range(len(lcd)):
            self.worksheet[f"B{start+i}"] = lcd[i]

        self.wb.save(self.path)

    def write_singleslice_thickness_report(self, thick_mean):

        #aggiornato al template nuovo
        print(f"[INFO] writing single slice thickness  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["C23"] = thick_mean
        self.wb.save(self.path)

    def write_cart_report(self, data):
        print(f"[INFO] writing cart  test results on {self.path} sheet: {self.sheet} ")
        #aggiornato
        self.worksheet["B116"] = data[0]
        self.worksheet["C116"] = data[1]
        self.worksheet["B117"] = data[2]
        self.worksheet["C117"] = data[3]

        self.wb.save(self.path)

    def write_iodine_report(self, results):

        #aggiornato nuovo template
        print(f"[INFO] writing iodine  test results on {self.path} sheet: {self.sheet} ")

        start=161
        for (i,r) in enumerate(results):

            self.worksheet[f"B{start+i}"] = r["center"]
            self.worksheet[f"C{start + i}"] = r["nord"]
            self.worksheet[f"D{start + i}"] = r["sud"]
            self.worksheet[f"E{start + i}"] = r["est"]
            self.worksheet[f"F{start + i}"] = r["ovest"]

        ##? BASELINE??


        self.wb.save(self.path)

    def write_resolution_catphan_report(self, metallic, background, noise, filter):
        # aggiornato al template

        if filter=="std":
            letter="B"
        elif filter=="bone":
            letter = "K"
        print(f"[INFO] writing contast resolution test results on {self.path} sheet: {self.sheet} ")

        self.worksheet[f"{letter}43"] = metallic
        self.worksheet[f"{letter}44"] = background

        start=48
        for (i,std) in enumerate(noise):
            self.worksheet[f"{letter}{start+i}"] = noise[i]

        self.wb.save(self.path)




