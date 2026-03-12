from run_scheduler import make_commands


commands = make_commands()

for i, cmd in enumerate(commands):
    print(f'[{i}/{len(commands)}] {cmd}')