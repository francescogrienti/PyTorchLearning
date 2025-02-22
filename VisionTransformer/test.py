import matplotlib.pyplot as plt

data = {"learning rate": 0.01, "batch size": 64, "epochs": 10}

# Creare una figura vuota
fig, ax = plt.subplots()

# Nascondere gli assi
ax.axis("off")

# Creare una tabella con i dati del dizionario
table_data = [[k, v] for k, v in data.items()]
table = ax.table(cellText=table_data, colLabels=["Parametro", "Valore"], loc="center")

plt.show()