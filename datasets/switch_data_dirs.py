import sys

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode not in ['cirrus', 'local']:
        raise ValueError("Invalid mode. Choose from 'cirrus' or 'local'.")

    with open('dataset_info.yaml', 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if 'root:' in line:
            if mode == 'cirrus':
                if '# Cirrus' in line:  # Uncomment Cirrus HPC root
                    if '# root:' in line:
                        lines[idx] = lines[idx].replace('# root:', 'root:')
                else:  # Comment out other roots
                    if '# root:' not in line:
                        lines[idx] = lines[idx].replace('root:', '# root:')
            elif mode == 'local':
                if '# Local' in line:  # Uncomment local root
                    if '# root:' in line:
                        lines[idx] = lines[idx].replace('# root:', 'root:')
                else:  # Comment out other roots
                    if '# root:' not in line:
                        lines[idx] = lines[idx].replace('root:', '# root:')

    with open('dataset_info.yaml', 'w') as f:
        f.writelines(lines)
