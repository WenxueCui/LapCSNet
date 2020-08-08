classdef vllab_dag_reshape < dagnn.ElementWise
    
    properties
        dims
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            sz = size(inputs{1}) ;
            inputs = inputs{1};
            dims = obj.dims;
            dims = horzcat(dims, 1) ;
            %     dims = horzcat(dims, size(x,4)) ;
            y = [];
            for i=1:sz(1)
                temp = [];
                for j=1:sz(2)
                    temp = [temp vl_nnreshape(inputs(i,j,:,:),dims)];
                end
                y = [y;temp];
            end
            outputs{1} = y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            sz = size(inputs{1}) ;
            derOutputs = derOutputs{1};
            dims = obj.dims;
            y = inputs{1};
            for i=1:sz(1)
                for j = 1:sz(2)
                    y(i,j,:,:) = vl_nnreshape(derOutputs((i-1)*dims(1)+1 : i*dims(1),...
                        (j-1)*dims(2)+1 : j*dims(2) , 1 , :), [1 1 dims(1)*dims(2)]);
                    
                end
            end
            derInputs{1} = y;
            derParams{1} = [];
        end
        
        
        
        function reset(obj)
            %obj.inputSizes = {} ;
        end
       
        
        function obj = vllab_dag_reshape(varargin)
            obj.load(varargin{:}) ;
			obj.dims = obj.dims ;
        end
    end
end



