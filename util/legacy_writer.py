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

        #mean values
        self.worksheet["C37"]=df["water_ct"].values[0]
        self.worksheet["C38"] = df["uniformity_up"].values[0]
        self.worksheet["C39"] = df["uniformity_bottom"].values[0]
        self.worksheet["C40"] = df["uniformity_right"].values[0]
        self.worksheet["C41"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D37"]=df["noise"].values[0]
        self.worksheet["D38"] = df["std_up"].values[0]
        self.worksheet["D39"] = df["std_bottom"].values[0]
        self.worksheet["D40"] = df["std_right"].values[0]
        self.worksheet["D41"] = df["std_left"].values[0]

        self.wb.save(self.path)


    def write_uniformity_head_report(self, df):
        print(f"[INFO] writing uniformity mode [HEAD] test results on {self.path} sheet: {self.sheet} ")
        #start changing values

        #mean values
        self.worksheet["C23"]=df["water_ct"].values[0]
        self.worksheet["C24"] = df["uniformity_up"].values[0]
        self.worksheet["C25"] = df["uniformity_bottom"].values[0]
        self.worksheet["C26"] = df["uniformity_right"].values[0]
        self.worksheet["C27"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D23"]=df["noise"].values[0]
        self.worksheet["D24"] = df["std_up"].values[0]
        self.worksheet["D25"] = df["std_bottom"].values[0]
        self.worksheet["D26"] = df["std_right"].values[0]
        self.worksheet["D27"] = df["std_left"].values[0]

        self.wb.save(self.path)

    def write_uniformity_monoenergetic_report(self,df):
        print(f"[INFO] writing uniformity mode [HEAD] test results on {self.path} sheet: {self.sheet} ")
        #start changing values

        #mean values
        self.worksheet["C140"]=df["water_ct"].values[0]
        self.worksheet["C141"] = df["uniformity_up"].values[0]
        self.worksheet["C142"] = df["uniformity_bottom"].values[0]
        self.worksheet["C143"] = df["uniformity_right"].values[0]
        self.worksheet["C144"] = df["uniformity_left"].values[0]

        #set noise values, namely the std
        self.worksheet["D140"]=df["noise"].values[0]
        self.worksheet["D141"] = df["std_up"].values[0]
        self.worksheet["D142"] = df["std_bottom"].values[0]
        self.worksheet["D143"] = df["std_right"].values[0]
        self.worksheet["D144"] = df["std_left"].values[0]

        self.wb.save(self.path)



    def write_uniformity_multislice_report(self, df,option):
        print(f"[INFO] writing uniformity mode [MULTISLICE] test results on {self.path} sheet: {self.sheet} ")
        #start changing values

        #mean values
        if option=="multislice":
            start_row=54
        elif option=="multislice_monoenergetic":
            start_row=156
        for i in range(len(df)):
            row=start_row+i
            self.worksheet[f"B{row}"]=df.loc[i,"water_ct"]
            self.worksheet[f"C{row}"] = df.loc[i,"uniformity_up"]
            self.worksheet[f"D{row}"] = df.loc[i,"uniformity_bottom"]
            self.worksheet[f"E{row}"] = df.loc[i,"uniformity_right"]
            self.worksheet[f"F{row}"] = df.loc[i,"uniformity_left"]



        self.wb.save(self.path)

    def write_linearity_report(self, lin):
        print(f"[INFO] writing linearity  test results on {self.path} sheet: {self.sheet} ")
        for i,val in enumerate(lin):
            self.worksheet[f"C{8+i}"] = val

        self.wb.save(self.path)

    def write_resolution_report(self, means, w, p, stds, std_w, std_p, filter):
        print(f"[INFO] writing resolution  test results on {self.path} sheet: {self.sheet} ")

        if filter=="std":
            start = 83
            self.worksheet[f"B78"] = p
            self.worksheet[f"B79"] = w

            self.worksheet[f"B88"] = w
            self.worksheet[f"C88"] = std_w

            self.worksheet[f"B89"] = p
            self.worksheet[f"C89"] = std_p

        elif filter=="bone":
            start=99
            self.worksheet[f"B94"] = p
            self.worksheet[f"B95"] = w

            self.worksheet[f"B104"] = w
            self.worksheet[f"C104"] = std_w

            self.worksheet[f"B105"] = p
            self.worksheet[f"C105"] = std_p



        for i, val in enumerate(means):
            self.worksheet[f"B{start + i}"] = means[i]
            self.worksheet[f"C{start + i}"] = stds[i]
        #self.worksheet[f"B{start + 5}"] = w
        #self.worksheet[f"B{start + 6}"] = p
        #self.worksheet[f"C{start + 5}"] = std_w
        #self.worksheet[f"C{start + 6}"] = std_p



        self.wb.save(self.path)

    def write_dot_thickness_report(self, fwhm):
        print(f"[INFO] writing linearity  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["C28"] = fwhm

        self.wb.save(self.path)

    def write_lowcontrast_report(self, mu_list,std_list, lcd):
        print(f"[INFO] writing lowcontrast  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["A112"] = mu_list[0]
        self.worksheet["B112"] = std_list[0]
        self.worksheet["C112"] = lcd[0]

        start=117
        for i in range(len(lcd)):
            self.worksheet[f"B{start+i}"] = lcd[i]

        self.wb.save(self.path)

    def write_singleslice_thickness_report(self, thick_mean):
        print(f"[INFO] writing single slice thickness  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["C29"] = thick_mean
        self.wb.save(self.path)

    def write_cart_report(self, data):
        print(f"[INFO] writing cart  test results on {self.path} sheet: {self.sheet} ")

        self.worksheet["B130"] = data[0]
        self.worksheet["C130"] = data[1]
        self.worksheet["B131"] = data[2]
        self.worksheet["C131"] = data[3]

        self.wb.save(self.path)

    def write_iodine_report(self, results):
        print(f"[INFO] writing iodine  test results on {self.path} sheet: {self.sheet} ")

        start=175
        for (i,r) in enumerate(results):

            self.worksheet[f"B{start+i}"] = r["center"]
            self.worksheet[f"C{start + i}"] = r["nord"]
            self.worksheet[f"D{start + i}"] = r["sud"]
            self.worksheet[f"E{start + i}"] = r["est"]
            self.worksheet[f"F{start + i}"] = r["ovest"]

        ##? BASELINE??


        self.wb.save(self.path)



