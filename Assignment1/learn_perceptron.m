function [w] = learn_perceptron(neg_examples_nobias,pos_examples_nobias)
    % sprintf("%s_examples = nx3 vector of %s examples (0.3, 0,4, 1)  <--bias", positive|negative)
    % w = 3x1 vector

    %Bookkeeping
    num_neg_examples = size(neg_examples_nobias,1);
    num_pos_examples = size(pos_examples_nobias,1);

    %Here we add a column of ones to the examples in order to allow us to learn
    %bias parameters.
    neg_examples = [neg_examples_nobias,ones(num_neg_examples,1)];
    pos_examples = [pos_examples_nobias,ones(num_pos_examples,1)];

    %Randomly initialse the Weight vector and the Generously Feasible Vector
    w = randn(3,1);

    %Find the data points that the perceptron has incorrectly classified
    [mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w);
    num_errs = size(mistakes0,1) + size(mistakes1,1);
    fprintf('Initial Number of errors :\t%d\n',num_errs);
    fprintf(['weights:\t', mat2str(w), '\n']);


    %Iterate until the perceptron has correctly classified all points.
    iter = 0
    while (num_errs > 0)
    	iter = iter + 1;

    	%Update the weights of the perceptron.
    	w = update_weights(neg_examples,pos_examples,w);


    	%Find the data points that the perceptron has incorrectly classified.
    	%and record the number of errors it makes.
    	[mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w);
    	num_errs = size(mistakes0,1) + size(mistakes1,1);


    	fprintf('Number of errors in iteration %d:\t%d\n',iter,num_errs);
    	fprintf(['weights:\t', mat2str(w), '\n']);
    	key = input('<Press enter to continue, q to quit.>', 's');
    	if (key == 'q')
    		break;
    	end
    end


    function [w] = update_weights(neg_examples, pos_examples, w_current)
        % WEARE DOING ONLINE TRAINING HERE, UPDATNG THE WEIGHT VECTOR AFTER PASS THROUGH EACH ELEMENT OF THE DATASET.
        % sprintf("%s_examples = nx3 vector of %s examples (0.3, 0,4, 1)  <--bias", positive|negative)
        % w = 3x1 vector

        w = w_current;
        num_neg_examples = size(neg_examples,1);
        num_pos_examples = size(pos_examples,1);

        for i=1:num_neg_examples
            current_input = neg_examples(i,:); %1x3
            output_of_perceptron = current_input*w;
            if (output_of_perceptron >= 0)
                w = w - current_input';
            end
        end

        for i=1:num_pos_examples
        	current_input = pos_examples(i,:); %1x3
        	output_of_perceptron = current_input*w;
        	if (output_of_perceptron < 0)
        		w = w + current_input';
        	end
        end


    function [mistakes0, mistakes1] =  eval_perceptron(neg_examples, pos_examples, w)
    	% sprintf("%s_examples = nx3 vector of %s examples (0.3, 0,4, 1)  <--bias", positive|negative)

    	num_neg_examples = size(neg_examples,1);
    	num_pos_examples = size(pos_examples,1);
    	mistakes0 = [];
    	mistakes1 = [];
    	for i=1:num_neg_examples
    		x = neg_examples(i,:); %x=1x3, w=3x1
    		output_of_perceptron = x*w;
    		if (output_of_perceptron >= 0)
    			mistakes0 = [mistakes0;i];
    		end
    	end
    	for i=1:num_pos_examples
    		x = pos_examples(i,:);
    		output_of_perceptron = x*w;
    		if (output_of_perceptron < 0)
    			mistakes1 = [mistakes1;i];
    		end
    	end
