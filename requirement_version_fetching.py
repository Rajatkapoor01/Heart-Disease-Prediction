def extractVersion():
    import pkg_resources

    # Input file with library names (without versions)
    input_file = 'lib.txt'
    # Output file with library names and their versions
    output_file = 'requirements.txt'
    
    # Read the library names from the input file
    with open(input_file, 'r') as file:
        libraries = [line.strip() for line in file.readlines()]
    
    # Open the output file to write the libraries and their versions
    with open(output_file, 'w') as output:
        for lib in libraries:
            try:
                # Get the installed version of the library
                version = pkg_resources.get_distribution(lib).version
                # Write the library and its version to the output file
                output.write(f'{lib}=={version}\n')
            except pkg_resources.DistributionNotFound:
                # If the library is not found, print an error message
                output.write(f'{lib}==<NOT INSTALLED>\n')
                print(f'Library "{lib}" is not installed.')
    
    print(f"Library versions have been written to {output_file}")
    
    
extractVersion()