import pandas as pd

if __name__ == '__main__':
    # df = pd.read_csv('/home/cindy/PycharmProjects/LLFlow/results/LOL-pc/measure_full_init.csv')
    df = pd.read_csv('/home/cindy/PycharmProjects/lowlight-baselines/LLFlow/results/LOL-pc/measure_lol.csv')

    print(df.head())
    print(f'psnr: {df["PSNR"].mean()}')
    print(f'ssim: {df["SSIM"].mean()}')
    print(f'lpips: {df["LPIPS"].mean()}')