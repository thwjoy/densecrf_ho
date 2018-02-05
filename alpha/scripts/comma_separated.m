function str = comma_separated(f, format)
if(f < 0), str = '-';
else    
    str = strtrim(sprintf(format, f));
    len = length(str);
%     for k = len-2:-3:2
%         str(k+1:end+1) = str(k:end);
%         str(k) = ',';
%     end
end
end
