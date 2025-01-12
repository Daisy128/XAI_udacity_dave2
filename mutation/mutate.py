from utils.mutation_utils import *
import mutation.utils.constants as const
from mutation.utils.logger_setup import setup_logger

def mutate_model(data):
    logger = setup_logger(__name__)

    file_path = data['subject_path'] # os.path.join('test_models', 'udacity_train_track1.py')
    model_name = data['subject_name'] # 'dave2'
    # list of mutations operators to be applied. Full list can be found in utils.constants
    mutations = data['mutations']
    root_dir = data['root_dir']
    props.model_name = model_name

    print("Root dir is: {}".format(root_dir))
    print("Model Name "+ model_name)
    print("Path " + file_path)

    mutation_types = ['D', 'H']

    mutants_path = os.path.join(str(root_dir), const.save_paths["mutated"], model_name)
    if not os.path.exists(mutants_path):
        os.makedirs(mutants_path)
        print("Created mutants directory: " + mutants_path)

    save_path_original = os.path.join(mutants_path, model_name + "_training_origin.py")

    prepare_model(file_path, save_path_original, mutation_types)

    for mutation in mutations:
        logger.info("Starting mutation %s", mutation)
        save_path_mutated = os.path.join(mutants_path, model_name + "_" + mutation + "_mutated")

        try:
            # Create mutationClass
            mutationClass = create_mutation(mutation)
            # call mutate, generate mutated model
            mutationClass.mutate(save_path_original, save_path_mutated)
        except LookupError as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Was not able to create a class for mutation %s: " + str(e), mutation)
        except Exception as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Unable to apply the mutation for mutation %s: " + str(e), mutation)


        logger.info("Finished mutation %s", mutation)

