import math

tolerances=input("Enter the tolerances (comma-separated): ")
print("Tolerances entered:", tolerances)

n_tolerances = len(tolerances.split(","))
sum=0

for tolerance in tolerances.split(","):
    tolerance = tolerance.strip()
    if not tolerance:
        continue

    try:
        print(f"Processing tolerance: {tolerance}")
        sum += (float(tolerance) / n_tolerances) ** 2
    except Exception as e:
        print(f"Error processing tolerance '{tolerance}': {e}")

sigma = round(math.sqrt(sum),5)
print("Calculated sigma:", sigma)

RSS = 3 * sigma
print("Calculated RSS: +-",RSS)

