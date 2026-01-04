import pandas as pd
import matplotlib.pyplot as plt

output_path = "/home/linux/mxm/output/experiments_backend/market1-1/1219/o-d/k_5/0/"
data = pd.read_csv(output_path + 'balance_ratio.csv')

mean_balance_ratio = data['balance_ratio'].mean()
std_balance_ratio = data['balance_ratio'].std()

print(f"Mean Balance Ratio: {mean_balance_ratio}")
print(f"Standard Deviation of Balance Ratio: {std_balance_ratio}")

plt.plot(data['global_frame_id'], data['balance_ratio'])
plt.xlabel('Frame ID')
plt.ylabel('Balance Ratio')
plt.title('Balance Ratio Trend')
plt.savefig(output_path + 'balance_ratio.png')
plt.show()
