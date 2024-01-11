import sys

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode not in ['hpc', 'local']:
        raise ValueError("Invalid mode. Choose from 'hpc' or 'local'.")

    with open('dataset_info.yaml', 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if 'root:' in line:
            if mode == 'hpc':
                # Uncomment HPC root
                if 'HPC' in line and '# root:' in line:
                    lines[idx] = lines[idx].replace('# root:', 'root:')
                # Comment out local root
                if 'Local' in line and '# root:' not in line:
                    lines[idx] = lines[idx].replace('root:', '# root:')
            elif mode == 'local':
                # Comment out HPC root
                if 'HPC' in line and '# root:' not in line:
                    lines[idx] = lines[idx].replace('root:', '# root:')
                # Uncomment local root
                if 'Local' in line and '# root:' in line:
                    lines[idx] = lines[idx].replace('# root:', 'root:')

    with open('dataset_info.yaml', 'w') as f:
        f.writelines(lines)
