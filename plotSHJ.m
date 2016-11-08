function plotSHJ(t, type)
    cats = size(t, 3);
    if cats == 3
        g = [1 3];
    else
        g = [ceil(sqrt(cats)), ceil(sqrt(cats))];
    end
    for c = 1:size(t, 3)
        hold on
        rotate3d on
        box on
        plot3(t(1, t(4, :, c) == 0, c) , t(2, t(4, :, c) == 0, c), t(3, t(4, :, c) == 0, c), 'ok', 'MarkerSize', 10, 'MarkerFaceColor', [0 0 0]);
        plot3(t(1, t(4, :, c) == 1, c) , t(2, t(4, :, c) == 1, c), t(3, t(4, :, c) == 1, c), 'ok', 'MarkerSize', 10, 'MarkerFaceColor', [1 1 1]);
        view([-30 -25])
        xlabel('Color');
        ylabel('Shape');
        zlabel('Size');
        title(['Category trained (T' num2str(type) ')']);
        set(gca, 'XTick', [])
        set(gca, 'YTick', [])
        set(gca, 'ZTick', [])
    end